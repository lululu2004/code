import pandas as pd
from datetime import datetime


def generate_report(student_id, name, score_list, fixed_answers):
    """
    生成并保存成绩单
    :param student_id: 学号
    :param name: 姓名
    :param score_list: 模型识别出的答案列表 ['A', 'C', ...]
    :param fixed_answers: 标准答案列表
    """
    # 计算得分（假设每题5分）
    total_score = sum([5 for i in range(len(score_list)) if score_list[i] == fixed_answers[i]])

    # 构建数据
    data = {
        '学号': [student_id],
        '姓名': [name],
        '总分': [total_score],
        '识别结果': ["".join(score_list)],
        '批改时间': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    }

    df = pd.DataFrame(data)

    # 导出为 Excel (如果文件存在则追加，不存在则新建)
    file_name = "class_results.xlsx"
    if not os.path.exists(file_name):
        df.to_excel(file_name, index=False)
    else:
        with pd.ExcelWriter(file_name, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            # 读取旧数据并合并，这里仅作示意
            old_df = pd.read_excel(file_name)
            new_df = pd.concat([old_df, df], ignore_index=True)
            new_df.to_excel(writer, index=False)

    print(f"成绩单已更新：{name} - {total_score}分")

# 调用示例：
# generate_report("2024001", "张三", ["A", "B", "C"], ["A", "D", "C"])