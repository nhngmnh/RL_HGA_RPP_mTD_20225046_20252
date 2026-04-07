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

- `nodes <= grid * grid` (với 10×10 thì `nodes <= 100`)
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
