#!/usr/bin/env python3
"""契约测试：动作输入掩码必须可配置，且训练/评测/部署显式声明同一机制。

纯 CPU、只用标准库，不需要 torch、GPU 或数据集：
  python -m unittest tests.test_action_masking_contract -v

检查点（对应第三轮审查的 5 个问题）：
  C1 modeling_prismatic.py 提供 set_action_input_masking / _resolve_action_input_masking
  C2 置零那一行确实被 mask_action_inputs 开关守住（AST 检查，不是文本搜索）
  C3 默认仍是 zero（OFT 并行/扩散解码行为不变）
  C4 forward 签名接受 mask_action_inputs 关键字
  C5 自回归训练脚本显式声明 keep，并在训练/评测前向里传 mask_action_inputs=False
  C6 部署侧缓存生成分支仍是 asserts labels is None + 真实 token 直通（即不置零），
     且前缀/回退前向显式传 mask_action_inputs=False
  C7 训练脚本没有自己重写置零逻辑
"""
import ast
import io
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MODELING = os.path.join(ROOT, "code", "openvla_dynamic_modules", "modeling_prismatic.py")
TRAIN = os.path.join(ROOT, "training", "train_pushT_pure2.py")
DEPLOY = os.path.join(ROOT, "code", "xpolicylab", "policy", "OpenVLA_Pure", "model.py")

CACHED_ASSERT = "Unexpected key `labels` provided during cached generation!"


def read(p):
    return io.open(p, encoding="utf-8").read()


@unittest.skipUnless(os.path.exists(MODELING), "modeling_prismatic.py not found")
class TestMaskingContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = read(MODELING)
        cls.tree = ast.parse(cls.src)

    def test_c1_helper_methods_exist(self):
        names = {n.name for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef)}
        self.assertIn("set_action_input_masking", names)
        self.assertIn("_resolve_action_input_masking", names)

    def test_c2_zeroing_is_guarded(self):
        """定位“把动作位置嵌入乘以 ~mask”的赋值，确认它在 if 里且条件调用了解析函数。"""
        found = []

        class V(ast.NodeVisitor):
            def visit_If(self, node):
                has_mult = False
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Assign):
                        for t in sub.targets:
                            if isinstance(t, ast.Name) and t.id == "input_embeddings":
                                v = sub.value
                                if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Mult):
                                    has_mult = True
                if has_mult:
                    found.append(ast.dump(node.test))
                self.generic_visit(node)

        V().visit(self.tree)
        self.assertTrue(found, "没有找到受 if 保护的动作位置置零赋值")
        self.assertTrue(any("_resolve_action_input_masking" in c for c in found),
                        "置零逻辑没有被 mask_action_inputs 开关守住")

    def test_c3_default_mode_is_zero(self):
        import re
        m = re.search(r'getattr\(self,\s*"action_input_masking",\s*"(\w+)"\)', self.src)
        self.assertIsNotNone(m, "无法确认默认 action_input_masking")
        self.assertEqual(m.group(1), "zero", "默认模式必须保持 zero 以兼容 OFT")
        self.assertIn('set_action_input_masking(self, mode="zero")', self.src)

    def test_c4_forward_accepts_kwarg(self):
        sigs = [n for n in ast.walk(self.tree)
                if isinstance(n, ast.FunctionDef) and n.name == "forward"]
        self.assertTrue(sigs)
        ok = False
        for s in sigs:
            args = [a.arg for a in s.args.args] + [a.arg for a in s.args.kwonlyargs]
            if "mask_action_inputs" in args:
                ok = True
        self.assertTrue(ok, "forward 签名缺少 mask_action_inputs")

    def test_c5_train_declares_keep(self):
        s = read(TRAIN)
        self.assertIn('set_action_input_masking("keep")', s)
        self.assertEqual(s.count("mask_action_inputs=False"), 2,
                         "训练与评测前向都应显式传 mask_action_inputs=False")
        self.assertIn('"action_input_masking": "keep"', s)

    def test_c6_deploy_keeps_real_tokens(self):
        s = read(DEPLOY)
        self.assertEqual(s.count("mask_action_inputs=False"), 2)
        # 部署侧缓存分支在 modeling 里断言 labels is None —— 说明它喂的是真实 token（不置零）
        self.assertIn(CACHED_ASSERT, self.src)
        self.assertIn('"action_input_masking": "keep"', s)

    def test_c7_train_does_not_reimplement_zeroing(self):
        """训练脚本不得自己重写置零逻辑（用 AST 查标识符，注释里的讨论不算）。"""
        tree = ast.parse(read(TRAIN))
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                used.add(n.attr)
        self.assertNotIn("all_actions_mask", used)
        self.assertNotIn("_process_action_masks", used)


if __name__ == "__main__":
    unittest.main(verbosity=2)
