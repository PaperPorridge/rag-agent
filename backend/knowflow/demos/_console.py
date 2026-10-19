"""
Windows 控制台编码适配 —— 所有 demo 的公共基建。

【这个文件修的坑】
  Windows 中文环境的控制台默认编码是 GBK（cp936），而 demo 输出里有
  ✅ ❌ → ↔ ¥ 这类字符。Python 往 GBK 控制台写这些字符时直接抛：

      UnicodeEncodeError: 'gbk' codec can't encode character '\\u2705'

  于是 demo 在第 1 个 emoji 处中断，后面的章节全跑不到 —— 表面上像代码有 bug，
  其实是输出编码没处理。

【为什么不直接把符号改成 [OK] / ->？】
  那会牺牲输出可读性，而且治标：下次有人加个新符号又炸。
  正确做法是让 stdout 用 UTF-8 写，并且**永不抛异常**。

【工程化理由】
  「我们的 demo 在任意 locale 下都能跑，因为入口统一做了 stdout 编码降级，
   并且 errors=backslashreplace 保证任何字符都不会让进程崩溃。」
  —— 生产服务同理：日志里出现异常字符不该把请求打挂。

【用法】
  def main() -> None:
      from knowflow.demos._console import safe_stdout
      safe_stdout()
      ...
"""

from __future__ import annotations

import sys


def safe_stdout(encoding: str = "utf-8") -> bool:
    """
    把 stdout/stderr 配成指定编码，且永不因编码错误崩溃。

    - encoding="utf-8"：现代 Windows Terminal / VS Code / macOS / Linux 都正常显示符号
    - errors="backslashreplace"：遇到真编不出的字符，写成 \\uXXXX 转义，绝不抛异常

    返回 True 表示配置成功（Python 3.7+ 才有 reconfigure）。
    """
    ok = False
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # Python < 3.7 或被替换成不支持的对象
            continue
        try:
            reconfigure(encoding=encoding, errors="backslashreplace")
            ok = True
        except (ValueError, OSError):
            # 极端情况：流已关闭 / 不可配置。不阻断 demo，交给默认行为。
            pass
    return ok


def install_excepthook() -> None:
    """
    兜底：即使 safe_stdout 漏了，未捕获异常也别因编码错误二次崩溃。
    traceback 里同样可能有非 ASCII 字符（文件名、变量值）。
    """
    original = sys.excepthook

    def _hook(exc_type, exc, tb):  # noqa: ANN001
        try:
            original(exc_type, exc, tb)
        except UnicodeEncodeError:
            sys.stderr.write(f"{exc_type.__name__}: {exc!r}\n")

    sys.excepthook = _hook
