# 🚀 Hybrid Genetic Algorithm with Reinforcement Learning for RPP-mTD

## 📌 Giới thiệu

Đồ án này tập trung vào việc **cài đặt lại và mở rộng** bài toán:

> **Rural Postman Problem with Multiple Trucks and Drones (RPP-mTD)**

được đề xuất trong bài báo:

> Sobhanan et al. (2025)

Mục tiêu chính của đồ án gồm hai phần:

1. **Reproduce (tái hiện)** thuật toán gốc:
   - Hybrid Genetic Algorithm (HGA)
   - Áp dụng cho bài toán phối hợp nhiều xe tải và drone

2. **Enhance (mở rộng)**:
   - Tích hợp **Reinforcement Learning (RL)** để cải thiện hiệu quả tìm kiếm của HGA

---

## 🎯 Bài toán

Cho một đồ thị vô hướng:

\[
G = (V, E)
\]

- Một tập cung bắt buộc:
\[
R \subseteq E
\]

- Một đội phương tiện:
  - \(K\) xe tải
  - Mỗi xe tải mang \(M\) drone

---

### 🎯 Mục tiêu

Tối thiểu hóa:

\[
\text{Makespan} = \max(\text{finish time của tất cả phương tiện})
\]

---

### ⚙️ Ràng buộc

- Drone phải:
  - Xuất phát và hạ cánh tại các node trên lộ trình của truck
- Giới hạn:
\[
\text{flight time} \le \tau
\]

- Ràng buộc δ-hop:
\[
\text{land\_hop} - \text{launch\_hop} \le \delta
\]

---

## 🧩 Sinh dataset (.txt) theo format URPP-like

Repo có hỗ trợ sinh instance `.txt` giống cấu trúc trong thư mục `dataset/`.

### Input

- Số node: `--nodes`
- Tổng số cạnh (undirected): `--edges`
- Số cạnh bắt buộc: `--required`

Mặc định:
- Lưới **10×10** (`--grid 10`)
- Tọa độ được sinh dạng **số thực** trên lưới tinh hơn với `--coord-scale` (mặc định 10 ⇒ bước 0.1)
- Sức chứa node tối đa: `(grid*coord-scale)^2` (vd: 10×10 với scale 10 ⇒ 10,000 nodes)
- Seed được cố định nội bộ và sẽ tự thay đổi theo index để mỗi file khác nhau nhưng vẫn tái lập được.

### Chi phí cạnh

- Cost của cạnh được tính theo **Manhattan distance** giữa 2 đỉnh:

  `|x_u - x_v| + |y_u - y_v|`

### Ràng buộc hợp lệ

- `nodes <= (grid*coord-scale)^2` (vd: 10×10 với scale 10 ⇒ tối đa 10,000 nodes)
- Để đồ thị liên thông: `edges >= nodes - 1`
- `edges <= nodes*(nodes-1)/2`
- `1 <= required <= edges`

### Cách chạy sinh dữ liệu trên map 100x100 mặc định

Ví dụ sinh instance theo convention folder/file:

```bash
python -m utils.dataset_generator --nodes 20 --edges 50 --required 20
```

Lệnh trên sẽ tự tạo (nếu chưa có):

- `dataset/N20/N20E50R20/N20E50R20_01.txt` (hoặc `_02`, `_03`... nếu đã tồn tại)

Nếu muốn dùng như hàm Python, xem `utils/dataset_generator.py`:

- `generate_urpp_grid_instance_text(...)`
- `write_urpp_grid_instance(...)`

---

## ▶️ Chạy thực nghiệm (main.py)

`main.py` hiện là entrypoint chạy thực nghiệm theo batch (mặc định chạy trên 5 instance và ghi kết quả ra `results.csv`).

### 1) Cài môi trường

Từ thư mục repo:

```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

### 2) Chuẩn bị dataset (đúng đường dẫn mà main.py đang load)

Mặc định `main.py` sẽ load các file:

- `dataset/N50/N50E200R50/N50E200R50_01.txt`
- ...
- `dataset/N50/N50E200R50/N50E200R50_05.txt`

Nếu chưa có, bạn có thể sinh đúng 5 file này bằng generator (chỉ cần chạy 5 lần với `--index` tương ứng):

```powershell
python -m utils.dataset_generator --nodes 50 --edges 200 --required 50 --index 1
python -m utils.dataset_generator --nodes 50 --edges 200 --required 50 --index 2
python -m utils.dataset_generator --nodes 50 --edges 200 --required 50 --index 3
python -m utils.dataset_generator --nodes 50 --edges 200 --required 50 --index 4
python -m utils.dataset_generator --nodes 50 --edges 200 --required 50 --index 5
```

### 3) Chạy

```powershell
python main.py
```

Trong quá trình chạy, chương trình sẽ in tiến trình và kết quả tốt nhất (makespan/fitness) cho từng instance.

### 4) Kết quả output

- Kết quả được **append** vào file `results.csv` ở thư mục gốc repo.
- Mỗi instance hiện ghi 2 dòng (HGA và RLHGA).
- Nếu muốn chạy lại từ đầu mà không cộng dồn, hãy đổi tên hoặc xoá `results.csv` trước khi chạy.
- Với RLHGA, Q-table của agent Q-learning sẽ được export ra CSV trong `outputs/qtable/` (mỗi instance 1 file).

### 5) Tuỳ biến nhanh

- Đổi danh sách instance / thư mục dataset: sửa vòng lặp trong `main.py` (đang chạy idx 1..5 cho bộ `N50E200R50`).
- Đổi tham số thuật toán: chỉnh trong `configs/algorithm_params.py` (ví dụ `G`, `PL`, `seed`).
- GA/RLGA hiện đang comment sẵn trong `main.py`; muốn chạy thêm thì bỏ comment các block tương ứng.
