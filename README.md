# OCR-PIPELINE

## Project target: Giải quyết bài toán tìm toạ độ từ ảnh sổ hồng/ sổ đỏ. Ghép thành một api hoàn chỉnh (ocr-pipeline) 

# Brief description:

	Từ ảnh sổ hồng/ sổ đỏ được cung cấp: 
	
	# Bước 1:
	Nhận ra vùng text và trích xuất text từ ảnh
	# Bước 2: 
	Từ kết quả các text thu được, detect và trích xuất ra được một cặp toạ độ chính xác nhất để xác định vị trí của bất động sản 
	# Bước 3:
	Sau khi đã có cặp toạ độ, chuyển toạ độ đó (gọi là toạ độ vn2k) thành kinh độ, vĩ độ trên Google Map 
	# Bước 4:
	Gọi api Google Map để thể hiện vị trí bất động sản trên Google Map  

# Explanation: 

	Lí do cần thể hiện chính xác toạ độ này là bởi vì khách hàng thường được dẫn dắt bởi các môi giới và không được biết vị trí chính xác của bất động sản đang cân nhắc giao dịch.
	Do đó, cần trích xuất và thể hiện đựoc vị trí của bất động sản theo sự thể hiện trên giấy tờ pháp lí như sổ hồng, sổ đỏ sẽ đem lại sự chính xác và minh bạch, rõ ràng trong giao dịch 

# Details: 

	# Bước 1: 
	Được thực hiện bởi model CRAFT và GDB_Scan (đoạn code được cung cấp trong ocr_test)
	# Bước 2: 
	Code cung cấp bởi Hưng và Duy (báo cáo kết quả nằm trong folder doc) - chạy tiếp ocr_process.ipnyb để xem phần hậu xử lí data
	# Bước 3:
	Sử dụng module của python : pyproj (hàm chuyển đổi toạ độ vn2k tìm được từ bước 2 thành lat-long theo vị trí Google Map) - code cung cấp trong folder vn2k_to_wgs84
	# Bước 4: 
	Gọi tới api của Google Map, parameters là cặp kinh độ , vĩ độ vừa nhận được từ bước ba  

# Test: 
	
	Mọi người test thử với các ảnh trong folder hình ảnh Hình ảnh sổ hồng-sổ đỏ nhé!
	File chính để chạy là ocr_process.ipnyb để test phần hậu xử lí data 

