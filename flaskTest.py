from flask import Flask, request, jsonify
import pymysql
import cv2
import numpy as np
import os
# 导入原有的选择题识别器和新增的 TrOCR 识别器
from detectSelect import ExamScanner
from numDetect_trocr import TrOCRDigitReader, preprocess_for_ocr_with_steps
import pandas as pd
app = Flask(__name__)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# --- 1. 数据库与模型初始化 ---
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '001128', # 请确保此处密码正确
    'database': 'grade_system',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# 保留你原有的选择题 TFLite 阅卷器
scanner = ExamScanner("handwritten_letter_model1.0.tflite")

# 全局初始化 TrOCR 识别器用于学号
print("正在初始化服务端 TrOCR 引擎...")
id_reader = TrOCRDigitReader()
TEMPLATE_PATH = "idTemplate.png" # 模板匹配用的图片

def get_db_conn():
    return pymysql.connect(**db_config)

# --- 2. 接口 A：学号识别 ---
@app.route('/predict_info', methods=['POST'])
def predict_info():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image"}), 400

    conn = None # 预先定义，防止 finally 报错
    try:
        img_bytes = request.files['image'].read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # --- OCR 识别部分 ---
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
        
        if template is None:
            return jsonify({"status": "error", "message": f"Template not found"}), 500
        
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val < 0.6:
            return jsonify({"status": "fail", "message": "未定位到学号区域"}), 200

        tw, th = template.shape[::-1]
        roi_x, roi_y = max_loc[0] + tw - 5, max_loc[1] - 5
        roi_bgr = img_bgr[roi_y : roi_y + th + 15, roi_x : roi_x + tw * 3 + 20]
        
        processed_roi = preprocess_for_ocr_with_steps(roi_bgr)
        student_id, confidence = id_reader.readtext(processed_roi)

        print(f"识别结果: {student_id}, 置信度: {confidence}")

        if not student_id:
            return jsonify({"status": "fail", "message": "识别出的学号为空"})

        # --- 数据库查询部分 (移动到第一个 try 内部或紧跟其后) ---
        conn = get_db_conn()
        with conn.cursor() as cursor:
            sql = "SELECT name, class_name FROM students WHERE student_id = %s"
            cursor.execute(sql, (student_id,))
            result = cursor.fetchone()


            
            if result:
                print(result['class_name'])
                return jsonify({
                    "status": "success",
                    "student_id": student_id,
                    "name": result['name'],
                    "class_name": result['class_name']
                })
            else:
                print("未知")
                return jsonify({
                    "status": "success", 
                    "student_id": student_id,
                    "name": "不在名单内",
                    "class_name": "未知"
                })

    except Exception as e:
        print(f"服务器内部错误: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if conn:
            conn.close()





    #     return jsonify({
    #         "status": "success",
    #         "student_id": student_id,
    #         "confidence": round(confidence, 4)
    #     })
    # except Exception as e:
    #     return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/query_student', methods=['POST'])
def query_student():
    sid = request.form.get('student_id')
    if not sid:
        return jsonify({"status": "error", "message": "Missing ID"})

    conn = get_db_conn()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT name, class_name FROM students WHERE student_id = %s"
            cursor.execute(sql, (sid,))
            result = cursor.fetchone()
            if result:
                print("rematch result:", result)
                return jsonify({
                    "status": "success",
                    "name": result['name'],
                    "class_name": result['class_name']
                })
            else:
                return jsonify({"status": "fail", "message": "Student not found"})
    finally:
        conn.close()


# --- 修改后的 predict 接口：只识别，不存库 ---
@app.route('/predict', methods=['POST'])
def predict():
    print(">>> 收到请求，开始处理图片...")
    try:
        
        file = request.files['image']
        cols = int(request.form.get('cols', 5))
        img_bytes = file.read()

        _, letters = scanner.process_and_score(
            img_bytes, 
            user_answers=None, 
            user_cols=cols
        )
        print(f">>> 返回数据: detected_letters = {letters}")
        return jsonify({
            "status": "success",
            "detected_letters": letters  # 把识别到的 A, B, C 返回给前端
        })
    except Exception as e:
        print(f"Error during recognition: {e}") 
        return jsonify({"status": "error", "message": f"识别失败: {str(e)}"}), 500

# --- 新增接口：接收前端确认后的数据并存库 ---
@app.route('/save_record', methods=['POST'])
def save_record():
    conn = None
    try:
        # 接收前端计算好的最终数据
        student_id = request.form.get('student_id')
        student_name = request.form.get('student_name')
        exam_name = request.form.get('exam_name')
        class_name = request.form.get('class_name')
        score = request.form.get('score')
        details = request.form.get('details') # 这是用户修改后的最终选项字符串

        if not all([student_id, score]):
            return jsonify({"status": "error", "message": "缺少必要参数"}), 400

        conn = get_db_conn()
        with conn.cursor() as cursor:
            sql = """INSERT INTO records (student_id, name, class_name, exam_name, score, details) 
                     VALUES (%s, %s, %s, %s, %s, %s)"""
            cursor.execute(sql, (student_id, student_name, class_name, exam_name, score, details))
        conn.commit()

        return jsonify({"status": "success", "message": "记录已保存"})
    except Exception as e:
        print(f"Save Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if conn:
            conn.close()
# --- 4. 辅助接口 (历史记录查询等) ---
@app.route('/get_history')
def get_history():
    conn = None
    try:
        class_name = request.args.get('class_name')
        exam_name = request.args.get('exam_name')
        conn = get_db_conn()

        with conn.cursor() as cursor:
                query = "SELECT * FROM records WHERE 1=1"
                params = []
                if class_name and class_name != "全部班级":
                    query += " AND class_name = %s"
                    params.append(class_name)
                if exam_name and exam_name != "全部考试":
                    query += " AND exam_name = %s"
                    params.append(exam_name)
                
                query += " ORDER BY create_time DESC"
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
    

        # 即使 rows 是空的，也要返回空数组 []
        return jsonify(rows)
    except Exception as e:
        print(f"Server Error: {e}") # 在电脑终端查看报错详情
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if conn is not None:
            conn.close()

@app.route('/get_classes')
def get_classes():
    conn = get_db_conn()
    try:
        # 使用 DictCursor 或普通 Cursor 均可，只要提取出字符串列表
        with conn.cursor() as cursor:
            # 从你的 classes 表中查询所有班级名称
            cursor.execute("SELECT class_name FROM classes")
            rows = cursor.fetchall() 
            
            # 将结果转换为简单的字符串列表: ["高一一班", "高一二班"]
            # 注意：根据你的数据库驱动，row 可能是 (name,) 或者是 {'class_name': name}
            class_names = []
            for row in rows:
                if isinstance(row, dict):
                    class_names.append(row['class_name'])
                else:
                    class_names.append(row[0])
            
            return jsonify(class_names) # 返回 JSON 数组
    except Exception as e:
        print(f"获取班级失败: {e}")
        return jsonify([]), 500
    finally:
        conn.close()

@app.route('/get_exams')
def get_exams():
    conn = get_db_conn()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT exam_name FROM exams")
            rows = cursor.fetchall()
            # 提取名称列表
            exams = [row['exam_name'] if isinstance(row, dict) else row[0] for row in rows]
        return jsonify(exams)
    finally:
        conn.close()

# 教务系统同步接口 (逻辑占位)
@app.route('/sync_to_school', methods=['POST'])
def sync_to_school():
    target_exam = request.form.get('exam_name')
    # 此处可编写将该考试成绩推送至学校教务系统的逻辑
    return jsonify({"status": "success", "message": f"考试 {target_exam} 已同步"})

@app.route('/import_students', methods=['POST'])
def import_students():
    file = request.files['file']
    df = pd.read_excel(file)
    # 遍历 df 并写入数据库的 students 表
    return jsonify({"status": "success"})

@app.route('/add_exam_single', methods=['POST'])
def add_exam_single():
    exam_name = request.form.get('exam_name')
    # 写入数据库的 exams 表
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)