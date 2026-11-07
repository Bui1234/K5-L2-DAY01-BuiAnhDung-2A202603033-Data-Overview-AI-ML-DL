# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 07/11/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python 3.13.18 / PyTorch 2.14.1+cpu / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không thay đổi code gốc của notebook. Chỉ thực hiện các thao tác chạy notebook, kiểm tra kết quả prediction và bổ sung nội dung báo cáo theo yêu cầu bài thực hành

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

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về toàn bộ bức ảnh, không phải một vật thể riêng biệt.
- Vì mô hình YOLO11n-cls thực hiện nhiệm vụ phân loại hình ảnh và đưa ra một nhãn đại diện cho nội dung tổng thể của bức ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Do đi kèm taxonmy/checkpoint
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- Vì mỗi thông tin có một chức năng khác nhau: 
`class_id`: xác định lớp
`class_name`: để con người dễ đọc và hiểu
`taxonomy_name`:xác định hệ thống phân loại mà lớp đó thuộc về 
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Guideline cần nêu rõ tiêu chí xác định chủ thể hoặc chủ đề chính đại diện cho toàn bộ bức ảnh là : tiêu trí lựa chọn, đối tượng chính và quy tắc ưu tiên và trường hợp đặc biệt

6. Vì sao không được lấy `score` của model làm ground truth?
- Vì score chỉ thể hiện mức độ tin cậy của mô hình đối với dự đoán, chứ không chứng minh dự đoán đó chính xác còn Ground Truth phải được xác định dựa trên nội dung thực tế của hình ảnh và quy trình gán nhãn có hướng dẫn, kiểm tra rõ ràng bởi con người.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= bowl ; `score`= 0.499948 ; `bbox_xyxy`="bbox_xyxy": [
      59.29,
      288.92,
      133.46,
      328.98
    ] ,
    `bbox_width`=74.18 ; `bbox_height`=40.06
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- Góc trên - trái: tọa độ (59.29, 288.92).
- Góc dưới - phải: tọa độ (133.46, 328.98).
- Chiều rộng: 74.18 pixel.
- Chiều cao: 40.06 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- Ngưỡng = 0.20: Có 17 dự đoán.
- Ngưỡng = 0.60: Có 6 dự đoán.
- Vậy khi tăng ngưỡng tin cậy từ 0.20 lên 0.60, số lượng dự đoán giảm từ 17 xuống 6. Nguyên nhân là các vật thể có điểm tin cậy thấp hơn ngưỡng mới sẽ bị loại bỏ.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Kiểm tra từng Bounding Box có xác định đúng vật thể không.
- Phát hiện các hộp bị dự đoán sai hoặc trùng lặp.
- Kiểm tra vị trí và kích thước của từng hộp.
- Loại bỏ các hộp không chính xác.
- Bổ sung hoặc chỉnh sửa nhãn nếu cần thiết.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Người gán nhãn phải vẽ khung Bounding Box bao quanh toàn bộ phần vật thể có thể nhìn thấy trong ảnh, đảm bảo khung sát với các cạnh của vật thể, không bao gồm quá nhiều vùng nền xung quanh và không bỏ sót phần vật thể nhìn thấy được.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Trong ảnh kitchen, có thể quan sát thấy một phần cánh tay/bàn tay của người ở sát mép trái ảnh bị cắt khỏi khung hình.
- Tôi sẽ đưa trường hợp này lên hỏi Lab Coach (escalation) nếu guideline chưa có hướng dẫn cụ thể về vật thể bị cắt mép. Vì cần xác nhận cách xử lý thống nhất đối với vật thể chỉ xuất hiện một phần trong ảnh, tránh việc mỗi người gán nhãn theo một cách khác nhau.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-011 ; `class_name`= spoon; `score`=0.359147 ; số điểm polygon=`polygon_point_count`=37 ; 3–5 điểm đầu của `polygon_xy`=   [529.0, 66.0 ], [ 529.0,72.0 ], [528.0, 73.0 ], [528.0, 74.0],
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- Polygon cung cấp thông tin chi tiết về đường biên và hình dạng thực tế của vật thể thông qua nhiều điểm tọa độ liên tiếp.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- `instance_id` là mã định danh riêng của từng đối tượng cụ thể trong ảnh, giúp phân biệt các vật thể khác nhau, kể cả khi chúng thuộc cùng một lớp.
- `instance_id` chỉ có chức năng phân biệt từng đối tượng trong dữ liệu phân đoạn, không xác định loại vật thể và không dùng để theo dõi chuyển động của vật thể qua các khung hình video.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- Người gán nhãn phải vẽ đường biên Mask sát theo đường viền thực tế của vật thể, bao phủ toàn bộ phần vật thể nhìn thấy được, không lấy thêm vùng nền xung quanh và không bao gồm các vật thể khác. Đối với những phần bị che khuất, chỉ vẽ phần vật thể có thể quan sát được.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Trong ảnh kitchen, có thể quan sát thấy các chiếc bát đặt gần nhau trên mặt bàn, khiến đường biên giữa một số chiếc bát khó xác định chính xác.
- Tôi sẽ sử dụng guideline nếu đã có hướng dẫn cụ thể về cách phân biệt các vật thể tiếp xúc hoặc chạm nhau. Nếu guideline chưa đề cập hoặc không thể xác định rõ đường biên giữa hai vật thể, tôi sẽ đưa trường hợp này lên escalation.
- Đảm bảo đường biên Mask được xác định thống nhất, tránh việc mỗi người gán nhãn theo một cách khác nhau.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn duy nhất đại diện cho toàn bộ bức ảnh. | Ảnh traffic có nhiều phương tiện nhưng mô hình dự đoán cab với score 0.510915. Khó xác định chủ thể chính.  |Xác định nội dung chính và chọn một nhãn phù hợp theo guideline.  | Kiểm tra nhãn có phản ánh đúng nội dung chính của ảnh không. |
| Phát hiện vật thể | Mỗi vật thể gồm một nhãn lớp và một Bounding Box (4 tọa độ). |Ảnh kitchen có prediction class , score , với bounding box. Các vật thể đặt gần nhau hoặc bị che khuất gây khó khăn khi xác định ranh giới bounding box. bowl 0.499948[59.29, 288.92, 133.46, 328.98]  | Vẽ Bounding Box bao quanh từng vật thể, chọn đúng nhãn và vị trí. | Kiểm tra nhãn, vị trí, kích thước hộp và các vật thể bị bỏ sót hoặc trùng lặp. |
| Instance segmentation | Mỗi vật thể gồm một nhãn lớp, một instance_id và một vùng Mask theo đường viền. | Ảnh kitchen có các vật thể đặt sát nhau, khiến đường biên giữa các vật thể khó xác định. |Vẽ Mask sát đường viền vật thể, phân biệt từng đối tượng và không lấy thừa nền.  |Kiểm tra đường biên Mask, vùng bị thiếu, vùng nền bị lấy thừa và việc phân biệt từng vật thể.  |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Chỉ sử dụng 3 ảnh COCO công khai được cung cấp trong bài Lab, không tải lên, chia sẻ hoặc sử dụng ảnh cá nhân và dữ liệu nội bộ khi chưa được cho phép.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Tôi sẽ dừng việc xử lý, không tải xuống hoặc chia sẻ dữ liệu đó, đồng thời báo ngay cho Lab Coach hoặc người phụ trách bài Lab để được hướng dẫn xử lý.
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
