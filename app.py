from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
import cv2
import numpy as np
import json

image_folder = "uploads"
output_file = "datas.json"

def write_json(question_number):

    all_results = {}
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png"))])

    for idx, image_name in enumerate(image_files):
        student_key = f"student{idx + 1}"
        image_path = os.path.join(image_folder, image_name)

        print(f"Processing {student_key} from {image_name}...")

        try:
            answers = optic_forms(question_number, image_path)
            all_results[student_key] = answers
        except Exception as e:
            print(f"{student_key} için hata oluştu: {e}")
            all_results[student_key] = {"error": str(e)}

    # JSON dosyasına yaz
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)


def optic_forms(question_number, image_path):


    # --- 1. Görüntüyü yükle ve ön işlemler ---
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Görüntüleri göster (orijinal, gri, kenarlar)
    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 3, 1)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title("Orijinal")
    # plt.axis("off")

    # plt.subplot(1, 3, 2)
    # plt.imshow(gray, cmap='gray')
    # plt.title("Gri")
    # plt.axis("off")

    # plt.subplot(1, 3, 3)
    # plt.imshow(edges, cmap='gray')
    # plt.title("Kenarlar")
    # plt.axis("off")
    # plt.show()

    # --- 2. Konturları bul ve en büyük dörtgeni seç ---
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    form_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            form_contour = approx
            break

    # Form kenarını çiz ve göster
    image_copy = image.copy()
    if form_contour is not None:
        cv2.drawContours(image_copy, [form_contour], -1, (0, 255, 0), 3)

    # plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    # plt.title("Form Kenarı")
    # plt.axis("off")
    # plt.show()

    # --- 3. Noktaları yeniden sırala (perspektif düzeltme için) ---
    def reorder_points(pts):
        pts = pts.reshape(4, 2)
        new_pts = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        new_pts[0] = pts[np.argmin(s)]  # top-left
        new_pts[2] = pts[np.argmax(s)]  # bottom-right
        diff = np.diff(pts, axis=1)
        new_pts[1] = pts[np.argmin(diff)]  # top-right
        new_pts[3] = pts[np.argmax(diff)]  # bottom-left
        return new_pts

    # --- 4. Perspektif düzeltme ---
    if form_contour is not None:
        pts = reorder_points(form_contour)
        (tl, tr, br, bl) = pts

        # Formun boyutlarını hesapla
        width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

        # Eğer form yatay (genişlik > yükseklik) ise, 90 derece döndür
        rotate_90 = width > height
        
        if rotate_90:
            # Yatay form için boyutları değiştir (90 derece döndürülmüş gibi)
            width, height = height, width
            # Noktaları 90 derece döndür: tl->tr, tr->br, br->bl, bl->tl
            pts = np.array([tr, br, bl, tl], dtype="float32")

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(image, matrix, (width, height))

        # plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        # plt.title(f"Düzeltilmiş Form {'(90° döndürülmüş)' if rotate_90 else ''}")
        # plt.axis("off")
        # plt.show()
    else:
        print("Form bulunamadı!")
        exit()

    # --- 5. Optik formun grid yapısını tanımla ---
    num_rows = question_number + 1  # 20 soru + başlık satırı olabilir
    num_cols = 5   # A, B, C, D, E şıkları

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = thresh.shape
    cell_height = h // num_rows

    # Sütun pozisyonlarını düzelt - toplam 6 sütun var (soru no + 5 şık)
    total_cols = num_cols + 1  # soru numarası sütunu dahil
    cell_width = w // total_cols

    # Satır ve sütun pozisyonlarını hesapla
    row_positions = np.linspace(0, h, num_rows + 1, dtype=int)
    # Tüm sütunları dahil et (soru no + A,B,C,D,E)
    col_positions = np.linspace(0, w, total_cols + 1, dtype=int)

    # --- 6. Hücreleri işaretle (görsel kontrol için) ---
    warped_marked = warped.copy()
    for row in range(num_rows):
        for col in range(total_cols):
            x1, x2 = col_positions[col], col_positions[col+1]
            y1, y2 = row_positions[row], row_positions[row+1]
            cv2.rectangle(warped_marked, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # plt.figure(figsize=(10, 20))
    # plt.imshow(cv2.cvtColor(warped_marked, cv2.COLOR_BGR2RGB))
    # plt.title("Grid Kutucukları")
    # plt.axis("off")
    # plt.show()

    # --- 7. İşaretli kutuları tespit et ---
    results = []

    for row in range(1, num_rows):  # 0.satır başlık, atla
        row_result = None
        max_fill = 0

        # Şık sütunları: 1=A, 2=B, 3=C, 4=D, 5=E (0 soru numarası)
        for col in range(1, total_cols):  # 1'den başla (soru no sütununu atla)
            x1, x2 = col_positions[col], col_positions[col+1]
            y1, y2 = row_positions[row], row_positions[row+1]

            # Hücrenin iç kısmını al (kenarları hariç tut)
            margin = 3  # Kenar boşluğu
            x1_inner, x2_inner = x1 + margin, x2 - margin
            y1_inner, y2_inner = y1 + margin, y2 - margin
            
            # Geçerli boyutları kontrol et
            if x2_inner <= x1_inner or y2_inner <= y1_inner:
                continue

            roi = warped[y1_inner:y2_inner, x1_inner:x2_inner]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            filled_pixels = cv2.countNonZero(thresh_roi)
            total_pixels = (x2_inner - x1_inner) * (y2_inner - y1_inner)
            
            # Doluluk oranını hesapla
            fill_ratio = filled_pixels / total_pixels if total_pixels > 0 else 0

            # Hem mutlak piksel sayısını hem de oranı kontrol et
            # Eşik değerlerini ayarla (boş formda çizgiler sorun çıkarmasın)
            min_pixels = 50  # Minimum piksel sayısı
            min_ratio = 0.15  # Minimum doluluk oranı (%15)
            
            if filled_pixels > max_fill and filled_pixels > min_pixels and fill_ratio > min_ratio:
                max_fill = filled_pixels
                # col=1 -> A (65), col=2 -> B (66), vb.
                row_result = chr(65 + (col - 1))

        results.append((row, row_result))

    # --- 8. Sonuçları yazdır ---
    # result = ""
    # for question_num, answer in results:
    #     result += f"Soru {question_num}: {answer if answer is not None else 'İşaretlenmemiş'}\n"

    # --- 8. Sonuçları yazdır ---
    result_dict = {}
    for row, answer in results:
        result_dict[str(row)] = answer if answer is not None else "None"

    return result_dict      


app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/upload', methods=['POST'])
def upload_image():
    # uploads klasörünü temizle
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f)
        if os.path.isfile(file_path):
            os.remove(file_path)


    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files.getlist['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    question_number = request.form.get('question_number', default=20)  # Varsayılan 20 soru
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # OpenCV ile işleme yapıldı.
    try:
        write_json(int(question_number))
    except Exception as e:
        return {'error': str(e)}

    return send_file("datas.json", as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render'ın verdiği PORT'u al
    app.run(host='0.0.0.0', port=port)