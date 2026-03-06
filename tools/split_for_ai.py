import os
import math
import re


def split_file_for_ai(file_path, output_dir="ai_context_parts", max_chars=8000):
    """
    将大文本文件拆分为 AI 友好的小片段，并根据“母文件名”自动生成输出 txt 的文件名前缀。

    设计目标（不改变原有拆分逻辑，只增强命名规则）：
    - 以母文件名（不含扩展名）作为前缀，例如：
        /path/to/20260202开发总结.md  ->  20260202开发总结_part_001.txt
    - 自动清洗不合法文件名字符，避免在 Windows/macOS/Linux 下写文件失败
    - 避免覆盖：同一输出目录中，如果同名前缀已存在，则自动追加 _v2/_v3...

    Args:
        file_path (str): 原始文件的绝对路径或相对路径。
        output_dir (str): 输出分割文件的目录。
        max_chars (int): 每个片段的最大字符数（建议 8000-10000 以确保安全）。

    保存位置说明：
        - 本脚本会把拆分后的 txt 文件保存到 output_dir 指定的目录中；
        - 默认 output_dir="ai_context_parts"（在当前工作目录下创建/使用该文件夹）。
    """

    if not os.path.exists(file_path):
        print(f"Error: 文件未找到: {file_path}")
        return

    # ---------- 1) 计算输出前缀（由母文件名决定） ----------
    # 例如 file_path = ".../20260202开发总结.md"
    # raw_prefix = "20260202开发总结"
    raw_prefix = os.path.splitext(os.path.basename(file_path))[0]
    prefix = sanitize_filename(raw_prefix)

    # 如果清洗后为空（极端情况：母文件名全是非法字符），给一个安全兜底
    if not prefix:
        prefix = "source"

    # ---------- 2) 准备输出目录 ----------
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 避免覆盖：如果 output_dir 中已经存在同样前缀的 part 文件，则改用 prefix_vN
    prefix = ensure_unique_prefix(output_dir, prefix)

    # ---------- 3) 读取源文件 ----------
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        print("Error: 文件编码读取失败，请确保文件是 UTF-8 格式。")
        return

    # ---------- 4) 预计算总字符数和预估分片数 ----------
    total_chars = sum(len(line) for line in lines)
    estimated_parts = math.ceil(total_chars / max_chars) if max_chars > 0 else 0

    current_part = 1
    current_chars = 0
    current_lines = []

    parts_created = []

    print(f"开始拆分文件... (总字符数: {total_chars}, 限制: {max_chars}/part)")
    print(f"输出文件名前缀: {prefix}")

    # ---------- 5) 按 max_chars 拆分（保持原有逻辑：按行累加，不拆行） ----------
    for line in lines:
        line_len = len(line)

        # 如果当前行加入后会超过限制，并且当前块不为空，则先保存当前块
        if current_chars + line_len > max_chars and current_lines:
            save_part(output_dir, prefix, current_part, estimated_parts, current_lines, parts_created)
            current_part += 1
            current_lines = []
            current_chars = 0

        current_lines.append(line)
        current_chars += line_len

    # 保存最后一个块
    if current_lines:
        save_part(output_dir, prefix, current_part, estimated_parts, current_lines, parts_created)

    print(f"\n拆分完成！共生成 {len(parts_created)} 个文件，位于: {os.path.abspath(output_dir)}")
    print(f"请按照文件名顺序（{prefix}_part_001.txt, {prefix}_part_002.txt ...）依次发送给 AI。")


def save_part(output_dir, prefix, part_num, total_estimated, lines, created_list):
    """
    保存单个片段，并添加 AI 上下文标记。

    Args:
        output_dir (str): 输出目录。
        prefix (str): 输出文件名前缀（由母文件名推导并清洗/去重）。
        part_num (int): 当前分片序号（从 1 开始）。
        total_estimated (int): 预估分片数量（保留参数，当前仅用于未来扩展；不影响功能）。
        lines (list[str]): 当前分片的文本行列表。
        created_list (list[str]): 已创建文件路径列表，用于统计与展示。
    """
    filename = f"{prefix}_part_{part_num:03d}.txt"
    filepath = os.path.join(output_dir, filename)

    header = f"--- [PART {part_num} START] (这是第 {part_num} 部分，请阅读并等待后续部分) ---\n\n"
    footer = f"\n\n--- [PART {part_num} END] ---"

    content = "".join(lines)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header + content + footer)

    created_list.append(filepath)
    print(f"已生成: {filename} ({len(content)} chars)")


def sanitize_filename(name):
    """
    将字符串清洗为“适合作为文件名的一部分”的形式。

    处理点：
    - 替换 Windows/macOS/Linux 常见非法字符：\\ / : * ? " < > |
    - 去掉首尾空白
    - 将连续下划线合并为单个下划线（更美观，非必须但推荐）

    Args:
        name (str): 原始名称（例如母文件名不含后缀）。

    Returns:
        str: 清洗后的名称。
    """
    if name is None:
        return ""

    # 1) 去掉首尾空白
    cleaned = name.strip()

    # 2) 替换非法字符为下划线
    cleaned = re.sub(r'[\\/:\\*\\?\\"<>\\|]+', "_", cleaned)

    # 3) 合并多个下划线为一个
    cleaned = re.sub(r"_{2,}", "_", cleaned)

    # 4) 再次去掉首尾下划线/空白（可选，但更整洁）
    cleaned = cleaned.strip(" _")

    return cleaned


def ensure_unique_prefix(output_dir, prefix):
    """
    确保在 output_dir 中使用该 prefix 不会导致覆盖已有文件。

    判定方式（简单且足够可靠）：
    - 如果存在 {prefix}_part_001.txt，则认为该 prefix 已被使用；
      于是改用 {prefix}_v2，再检查 {prefix}_v2_part_001.txt；
      依此类推直到找到未占用的前缀。

    Args:
        output_dir (str): 输出目录。
        prefix (str): 期望使用的前缀。

    Returns:
        str: 不会覆盖的唯一前缀。
    """
    candidate = prefix
    version = 1

    while True:
        test_name = f"{candidate}_part_001.txt"
        test_path = os.path.join(output_dir, test_name)
        if not os.path.exists(test_path):
            return candidate
        version += 1
        candidate = f"{prefix}_v{version}"


if __name__ == "__main__":
    # ================= 配置区域 =================
    #
    # 请在这里输入您要拆分的文档路径（支持绝对路径/相对路径）
    # 例如:
    #   TARGET_FILE = r"D:\\Program\\Anla\\Logs\\20260202开发总结.md"
    #
    # 保存位置（默认）:
    #   拆分后的文件将保存到当前目录下的 ai_context_parts/ 目录中
    #
    # ===========================================

    TARGET_FILE = "conversation_history.txt"  # <--- 修改这里为您的大文件名/路径

    # 如果文件不存在，则提示用户输入
    if not os.path.exists(TARGET_FILE):
        user_input = input("请输入要拆分的文件路径: ").strip()
        # 去除可能存在的引号，避免复制路径时带引号导致找不到文件
        TARGET_FILE = user_input.replace('"', "").replace("'", "")

    split_file_for_ai(TARGET_FILE)
