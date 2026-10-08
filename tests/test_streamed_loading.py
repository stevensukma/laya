"""Target-device checkpoint loading regressions; tiny local weights, no network.

Run: python tests/test_streamed_loading.py
CUDA placement tests run when a CUDA device is available; CPU tests exercise the
same loader, metadata validation, precision conversion and public API forwarding.
"""
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import weakref
from types import SimpleNamespace

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import (AutoModel, BertConfig, ModernBertConfig, PreTrainedTokenizerFast)  # noqa: E402

from laya import Agent, Router, load  # noqa: E402
from laya.agent import _load_weights_on_device  # noqa: E402
from laya.common import DecisionModel, build_model  # noqa: E402


class StreamedLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.repos = {}
        cls.cfg = {"encoder": "unused/offline", "head_layers": 1, "act_costs": {"act": 0},
                   "max_len": 64, "head_max_len": 32, "temperature": [1.1, 1.2, 1.3]}
        cls.questions = {
            "choice": {"type": "choice", "instructions": "Pick one", "criteria": ["yes", "no"]},
            "flag": {"type": "noul", "instructions": "hello"},
            "level": {"type": "score", "instructions": "Rate hello", "criteria": [0.0, 1.0]},
        }
        for family, config in (
            ("bert", BertConfig(vocab_size=7, hidden_size=64, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=128)),
            ("modernbert", ModernBertConfig(vocab_size=7, hidden_size=64, intermediate_size=128,
                                          num_hidden_layers=1, num_attention_heads=2,
                                          max_position_embeddings=64, local_attention=8, pad_token_id=0)),
        ):
            root = Path(cls.tmp.name) / family
            root.mkdir()
            config.save_pretrained(root / "encoder")
            tokens = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
                                         "[MASK]": 4, "hello": 5, "yes": 6}, unk_token="[UNK]"))
            tokens.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokens, pad_token="[PAD]",
                                                unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]",
                                                mask_token="[MASK]")
            tokenizer.save_pretrained(root / "tokenizer")
            model = DecisionModel(AutoModel.from_config(config, attn_implementation="sdpa"), head_layers=1)
            # The model stays float32 even when the checkpoint stores half precision.
            save_file({name: value.half() if value.is_floating_point() else value
                       for name, value in model.state_dict().items()}, root / "model.safetensors")
            (root / "rl_agent_config.json").write_text(json.dumps(cls.cfg))
            cls.repos[family] = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_cpu_weights_buffers_predictions_and_rng_match_legacy(self):
        for family, root in self.repos.items():
            with self.subTest(family=family):
                legacy = Agent(str(root), device="cpu")
                rng = torch.get_rng_state().clone()
                streamed = Agent(str(root), device="cpu")
                with patch("safetensors.torch.load_file", side_effect=AssertionError("Full checkpoint read")):
                    streamed.model = build_model(self.cfg, str(root / "encoder"), pretrained=False, device="cpu")
                    _load_weights_on_device(streamed.model, self.cfg, str(root / "model.safetensors"),
                                            torch.device("cpu"), str(root))
                    streamed.model.eval()
                    streamed.model.encoder.config.reference_compile = False
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual({p.dtype for p in streamed.model.parameters()}, {torch.float32})
                for name, value in legacy.model.state_dict().items():
                    torch.testing.assert_close(streamed.model.state_dict()[name], value, rtol=0, atol=0)
                for name, value in legacy.model.named_buffers():
                    torch.testing.assert_close(dict(streamed.model.named_buffers())[name], value, rtol=0, atol=0)
                if family == "modernbert":
                    self.assertTrue(any("inv_freq" in name for name, _ in streamed.model.named_buffers()))
                self.assertEqual(streamed.system_one("hello", self.questions),
                                 legacy.system_one("hello", self.questions))
                self.assertEqual(streamed.temperature, legacy.temperature)

    def test_checkpoint_mismatch_fails_before_any_tensor_is_read(self):
        root = self.repos["bert"]
        weights = load_file(root / "model.safetensors")
        cases = {
            "missing parameter": {name: value for name, value in weights.items() if name != "scorer.1.weight"},
            "unexpected key": dict(weights, extra=torch.ones(1)),
            "bad parameter shape": dict(weights, **{"scorer.1.weight": torch.zeros(1)}),
            "missing buffer": {name: value for name, value in weights.items() if name != "temperature"},
            "bad buffer shape": dict(weights, temperature=torch.ones(4)),
        }
        for case, invalid in cases.items():
            with self.subTest(case=case):
                path = Path(self.tmp.name) / "invalid.safetensors"
                save_file(invalid, path)
                model = build_model(self.cfg, str(root / "encoder"), pretrained=False)
                reads = []
                with safe_open(path, framework="pt", device="cpu") as actual:
                    class MetadataOnly:
                        def __enter__(self):
                            return self

                        def __exit__(self, *args):
                            pass

                        def keys(self):
                            return actual.keys()

                        def get_slice(self, name):
                            return actual.get_slice(name)

                        def get_tensor(self, name):
                            reads.append(name)
                            raise AssertionError("Read before validation")

                    with patch("safetensors.safe_open", return_value=MetadataOnly()):
                        with self.assertRaises(ValueError):
                            _load_weights_on_device(model, self.cfg, str(path), torch.device("cpu"), str(root))
                self.assertEqual(reads, [])

    def test_one_tensor_is_released_before_the_next_read(self):
        root = self.repos["bert"]
        model = build_model(self.cfg, str(root / "encoder"), pretrained=False)
        previous = []
        with safe_open(root / "model.safetensors", framework="pt", device="cpu") as actual:
            class TrackingCheckpoint:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def keys(self):
                    return actual.keys()

                def get_slice(self, name):
                    return actual.get_slice(name)

                def get_tensor(self, name):
                    gc.collect()
                    if previous:
                        if previous[-1]() is not None:
                            raise AssertionError("Previous checkpoint tensor is still held")
                    tensor = actual.get_tensor(name)
                    previous.append(weakref.ref(tensor))
                    return tensor

            with patch("safetensors.safe_open", return_value=TrackingCheckpoint()):
                _load_weights_on_device(model, self.cfg, str(root / "model.safetensors"),
                                        torch.device("cpu"), str(root))
        self.assertEqual(len(previous), len(model.state_dict()))
        self.assertTrue(all(ref() is None for ref in previous))

    def test_agent_automatically_streams_only_when_cuda_is_resolved(self):
        import laya.agent as agent_module
        root = str(self.repos["bert"])
        with patch.object(agent_module, "build_model", wraps=build_model) as build:
            Agent(root, device="cpu")
            self.assertNotIn("device", build.call_args.kwargs)
        # Exercise CUDA selection without requiring CUDA hardware. The real CUDA
        # placement and buffer checks below run separately on a CUDA-capable host.
        model = build_model(self.cfg, str(self.repos["bert"] / "encoder"), pretrained=False)
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.device_count", return_value=2), \
                patch("torch.cuda.get_device_capability", return_value=(8, 0)), \
                patch.object(agent_module, "build_model", return_value=model) as build, \
                patch.object(agent_module, "_load_weights_on_device") as stream, \
                patch.object(model, "to", return_value=model), \
                patch("safetensors.torch.load_file", side_effect=AssertionError("Full checkpoint read")):
            agent = load(root, device="cuda:1")
        self.assertEqual(agent.device, torch.device("cuda:1"))
        self.assertEqual(build.call_args.kwargs["device"], torch.device("cuda:1"))
        self.assertEqual(stream.call_args.args[3], torch.device("cuda:1"))
        self.assertEqual(stream.call_count, 1)

    def test_cuda_unavailable_keeps_existing_cpu_loading_policy(self):
        with patch("torch.cuda.is_available", return_value=False), \
                patch("laya.agent._load_weights_on_device", side_effect=AssertionError("Unexpected streaming")):
            with self.assertWarns(RuntimeWarning):
                agent = load(str(self.repos["bert"]), device="cuda")
        self.assertEqual(agent.device, torch.device("cpu"))

    def test_unindexed_and_auto_cuda_use_the_current_gpu(self):
        for requested in ("cuda", "auto", None):
            with self.subTest(device=requested):
                model = build_model(self.cfg, str(self.repos["bert"] / "encoder"), pretrained=False)
                with patch("torch.cuda.is_available", return_value=True), \
                        patch("torch.cuda.device_count", return_value=2), \
                        patch("torch.cuda.current_device", return_value=1), \
                        patch("torch.cuda.get_device_capability", return_value=(8, 0)), \
                        patch("laya.agent.build_model", return_value=model) as build, \
                        patch("laya.agent._load_weights_on_device") as stream, \
                        patch.object(model, "to", return_value=model), \
                        patch("safetensors.torch.load_file", side_effect=AssertionError("Full checkpoint read")):
                    agent = load(str(self.repos["bert"]), device=requested)
                self.assertEqual(agent.device, torch.device("cuda:1"))
                self.assertEqual(build.call_args.kwargs["device"], torch.device("cuda:1"))
                self.assertEqual(stream.call_args.args[3], torch.device("cuda:1"))

    def test_router_forwards_cuda_without_an_extra_loading_option(self):
        with patch("laya.agent.Agent") as constructor:
            router = Router(device="cuda", models={"tiny": str(self.repos["bert"])})
            router.preload(["tiny"])
        self.assertEqual(constructor.call_args.kwargs["device"], "cuda")

    def test_streamed_allocation_errors_propagate_without_cpu_retry(self):
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.device_count", return_value=1), \
                patch("laya.agent.build_model", side_effect=torch.cuda.OutOfMemoryError("test")) as build:
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                Agent(str(self.repos["bert"]), device="cuda:0")
        self.assertEqual(build.call_count, 1)

    def test_cuda_copy_is_synchronized_before_reading_next_tensor(self):
        root = self.repos["bert"]
        model = build_model(self.cfg, str(root / "encoder"), pretrained=False)
        events = []
        state = {name: SimpleNamespace(shape=value.shape, device=torch.device("cuda:1"),
                                      copy_=lambda tensor, name=name: events.append(("copy", name)))
                 for name, value in model.state_dict().items()}
        stream = Mock()
        stream.synchronize.side_effect = lambda: events.append(("sync", None))
        with safe_open(root / "model.safetensors", framework="pt", device="cpu") as actual:
            class TrackingCheckpoint:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def keys(self):
                    return actual.keys()

                def get_slice(self, name):
                    return actual.get_slice(name)

                def get_tensor(self, name):
                    events.append(("read", name))
                    return actual.get_tensor(name)

            with patch("safetensors.safe_open", return_value=TrackingCheckpoint()) as opened, \
                    patch.object(model, "state_dict", return_value=state), \
                    patch("torch.cuda.current_stream", return_value=stream) as current_stream:
                _load_weights_on_device(model, self.cfg, str(root / "model.safetensors"),
                                        torch.device("cuda:1"), str(root))
        self.assertEqual(opened.call_args.kwargs["device"], "cuda:1")
        self.assertEqual(current_stream.call_count, len(state))
        self.assertTrue(all(call.args == (torch.device("cuda:1"),) for call in current_stream.call_args_list))
        self.assertEqual(events, [event for name in state
                                  for event in (("read", name), ("copy", name), ("sync", None))])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
    def test_cuda_parameters_and_runtime_buffers_match_cpu(self):
        for family, root in self.repos.items():
            for index in range(torch.cuda.device_count()):
                with self.subTest(family=family, device=index):
                    cpu = Agent(str(root), device="cpu")
                    cuda = Agent(str(root), device=f"cuda:{index}")
                    self.assertEqual(cuda.device, torch.device(f"cuda:{index}"))
                    self.assertEqual({p.device for p in cuda.model.parameters()}, {cuda.device})
                    for name, value in cpu.model.state_dict().items():
                        torch.testing.assert_close(cuda.model.state_dict()[name].cpu(), value, rtol=0, atol=0)
                    for name, value in cpu.model.named_buffers():
                        gpu_value = dict(cuda.model.named_buffers())[name]
                        self.assertEqual(gpu_value.device, cuda.device)
                        torch.testing.assert_close(gpu_value.cpu(), value, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
