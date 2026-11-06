# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:6/11/2026**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
`class_id`=577 ; `class_name`=gong ; `rank`=1 ; `score`=0.420192 ; `taxonomy_name`=ImageNet-1K
`class_id`=762 ; `class_name`=restaurant ; `rank`=1 ; `score`=0.792161; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record trong dữ liệu này nói về toàn bộ bức ảnh, không phải một vật thể riêng lẻ.
Vì:
- Mỗi sample_id đại diện cho một bức ảnh.
- Mỗi ảnh có danh sách Top-5 lớp mà mô hình dự đoán.
- Không có thông tin tọa độ bounding box hoặc vùng chứa vật thể cụ thể.
- Mô hình đưa ra một lớp dự đoán đại diện cho nội dung tổng thể của ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách lớp được xác định bởi taxonomy và đi kèm với mô hình đã huấn luyện (checkpoint), không phải do mô hình tự nghĩ ra.
Trong JSON:
- model_file: yolo11n-cls.pt – checkpoint mô hình phân loại.
- taxonomy_name: ImageNet-1K – bộ danh mục lớp.
- class_id: Mã định danh của lớp.
- class_name: Tên lớp tương ứng.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Cần ghi cả ba trường để xác định lớp chính xác, tránh nhầm lẫn khi sử dụng nhiều bộ phân loại khác nhau.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần quy định rõ tiêu chí chọn lớp đại diện cho toàn bộ bức ảnh, bao gồm:
- Ưu tiên chủ thể chính hoặc đối tượng chiếm vai trò quan trọng nhất trong ảnh.
- Nếu có nhiều đối tượng, chọn đối tượng nổi bật nhất theo mục tiêu của bộ dữ liệu.
- Nếu không xác định được chủ thể chính, cần có quy tắc ưu tiên hoặc nhãn phù hợp như Other hoặc Unknown nếu taxonomy hỗ trợ.
- Tất cả người gán nhãn phải áp dụng cùng một tiêu chí để đảm bảo tính nhất quán.
6. Vì sao không được lấy `score` của model làm ground truth?
Vì score chỉ thể hiện mức độ tin cậy tương đối của mô hình đối với dự đoán, không chứng minh dự đoán đó đúng.
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Góc trên-trái: tọa độ (385.33, 69.24).
Góc dưới-phải: tọa độ (498.92, 348.92).
Chiều rộng: 113.58 pixel.
Chiều cao: 279.68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Chọn hai ngưỡng 0.35 và 0.60.
- Ngưỡng 0.35: 11 prediction.
- Ngưỡng 0.6: 6 prediction.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Khi hạ ngưỡng, mô hình giữ lại nhiều hộp hơn nên reviewer cần:
- Kiểm tra các đối tượng được phát hiện.
- Loại bỏ những bounding box không đúng.
- Kiểm tra các đối tượng bị nhận diện nhầm.
- Phát hiện các bounding box trùng lặp.
- Xác nhận nhãn và vị trí của từng đối tượng.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Người gán nhãn phải vẽ hộp bao quanh toàn bộ phần nhìn thấy của đối tượng, đảm bảo khung sát với đường viền đối tượng, không lấy quá nhiều vùng nền xung quanh và không bỏ sót phần đối tượng có thể quan sát được.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Chọn : người sát mép bên trái(bị cát mép, còn một bàn tay xuất hiện)
- Đưa lên hỏi Lab Coach vì đối tượng không xuất hiện đầy đủ trong ảnh nên cần hỏi Lab Coach có nên gắn nhãn hay không, nếu có thì vẽ box thế nào.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record:
instance_id=`kitchen-001` ; class_name=`person` ; score=`0.899318` ; số điểm polygon =`polygon_point_count`=`348` ; 5 điểm đầu của polygon_xy=`[(446.0, 70.0), (445.0, 71.0), (444.0, 71.0), (443.0, 72.0), (442.0, 72.0)]`
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- Box chỉ cho biết khung chữ nhật bao quanh vật thể; polygon mô tả đường biên và hình dạng thực tế sát hơn, kể cả phần lồi lõm.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- instance_id là mã riêng để phân biệt từng vật thể được gán nhãn trong mẫu, kể cả khi nhiều vật cùng lớp. Nó không cho biết lớp của vật (đó là class_id/`class_name`) hay theo dõi vật đó qua các khung hình (tracking ID).
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- Vẽ biên sát theo phần vật thể nhìn thấy; không bao gồm nền hoặc phần bị vật khác che khuất, và không tự đoán phần khuất.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Với kitchen-002 (bowl), box nằm khoảng x=31–101, y=344–386 và chồng vùng box của bàn ăn. Đây là vị trí có thể cần kiểm tra ranh giới bát với mặt bàn. Chỉ từ file tọa độ, không thể xác định chính xác độ mờ trên ảnh; nếu biên nhìn thấy rõ thì áp dụng guideline, còn nếu bát và bàn chạm nhau khiến biên không thể xác định chắc chắn thì dừng và hỏi người hướng dẫn để escalation.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |  |  |  |  |
| Phát hiện vật thể |  |  |  |  |
| Instance segmentation |  |  |  |  |
- Phân loại ảnh | Một nhãn lớp cho toàn ảnh | Ảnh kitchen được dự đoán hạng 1 là gong, score 0.420192 (JSON), dù PNG cho thấy cảnh bếp với nhiều nồi, chảo và đồ dùng. Cần kiểm tra theo tiêu chí chọn lớp của bài toán. | Chọn lớp đại diện cho toàn ảnh theo guideline; hỏi người hướng dẫn nếu tiêu chí chưa rõ. | Nhãn có phản ánh nội dung toàn ảnh và đúng taxonomy không; không xem score là ground truth.
- Phát hiện vật thể | Một hộp và một nhãn lớp cho mỗi vật thể | PNG/JSON cho thấy người ở sát mép trái bị cắt. Detection JSON ghi person, box [0.12, 263.18, 61.55, 310.78], score 0.610931. Cần biết guideline xử lý vật thể bị cắt mép thế nào. | Gán box cho phần nhìn thấy theo guideline; hỏi người hướng dẫn nếu chưa rõ có nên gán vật thể bị cắt mép không. | Kiểm tra lớp, vị trí và độ sát của box; xem có vật thể bị bỏ sót hoặc box trùng không.
- Instance segmentation | Một nhãn lớp và một mask/polygon cho mỗi vật thể riêng | Trong PNG, mask dining table có vẻ phủ cả một số đồ vật trên mặt bàn. JSON ghi kitchen-005, box [0.97, 234.74, 345.86, 423.87], polygon có 598 điểm. Cần kiểm tra mask có lấn sang vật thể riêng không. | Vẽ mask theo phần nhìn thấy của từng vật; tách vật chạm nhau và hỏi người hướng dẫn khi biên khuất hoặc không rõ. | Kiểm tra mask có bám biên, lấn sang đồ vật/nền, hoặc gộp nhầm các instance không.

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Chỉ thu thập và sử dụng dữ liệu cần thiết, đúng phạm vi được cho phép trong bài lab, đồng thời bảo mật dữ liệu và không chia sẻ ra ngoài.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng lại và báo ngay cho giảng viên hoặc người hướng dẫn phụ trách bài lab
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
