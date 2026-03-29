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
