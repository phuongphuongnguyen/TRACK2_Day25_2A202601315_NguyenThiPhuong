# Lab 25 — Tối ưu hóa Chi phí GPU (GPU FinOps Workshop)

> **AICB · Phase 2 · Track 2 (Infrastructure) · Day 25**
> Kết hợp với slide `day25-gpu-finops-cost-optimization`.
> Đầu ra chính: **Báo cáo chi phí GPU (baseline vs. optimized)** — đầu vào cho Milestone 2.

---

## Bối cảnh

Bạn vừa được tuyển làm **FinOps Engineer** tại *NimbusAI* — một startup LLM đang có hóa đơn GPU mất kiểm soát. Bạn được giao:

- Dữ liệu telemetry GPU tổng hợp (thực tế)
- Danh mục giá GPU tháng 6/2026
- Nhật ký sử dụng token

**Nhiệm vụ:** Tìm ra điểm lãng phí và **cắt giảm chi phí 40–95%** — đo bằng `$/1M-token`, không phải `$/GPU-giờ`.

> Lab này chạy hoàn toàn trên **laptop không có GPU, không có tài khoản cloud, không cần API key**.

---

## Cài đặt môi trường

```bash
cd Day25-Track2-GPU-FinOps-Lab

# Tạo và kích hoạt virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt

# Kiểm tra cài đặt (phải in ra 11/11 checks passed)
python verify.py
```

### Chạy toàn bộ lab nhanh

```bash
python data/generate.py       # (Tạo lại) dữ liệu tổng hợp (seed cố định = 25)
python missions/run_all.py    # Chạy M1 → M5, in kết quả
pytest -q                     # Chạy 15 unit + integration tests
```

Kết quả xuất ra thư mục `outputs/`: `report.md`, `savings.png`, `focus_export.csv`.

---

## Cấu trúc dự án

```
Day25-Track2-GPU-FinOps-Lab/
├── data/
│   ├── generate.py           # Sinh dữ liệu tổng hợp (tất định, seed=25)
│   ├── price_catalog.csv     # Bảng giá 7 loại GPU
│   ├── gpu_telemetry.csv     # Telemetry 11 GPU × 24 giờ
│   ├── token_usage.csv       # 2,400 request LLM
│   └── workloads.csv         # 8 workload training/inference
├── finops/
│   ├── metrics.py            # MFU, MBU, roofline, phát hiện "GPU-Util lie"
│   ├── pricing.py            # Chi phí request, $/1M-token, chiết khấu, tier
│   ├── allocation.py         # Phân bổ chi phí theo tag, FOCUS export
│   ├── sustainability.py     # Năng lượng, carbon, vùng tối ưu
│   └── report.py             # Tạo báo cáo Markdown + biểu đồ
├── missions/
│   ├── m1_efficiency_audit.py   # Mission 1: Kiểm toán hiệu quả GPU
│   ├── m2_inference_levers.py   # Mission 2: Đòn bẩy chi phí inference
│   ├── m3_purchasing.py         # Mission 3: Chiến lược mua GPU
│   ├── m4_allocation.py         # Mission 4: Phân bổ chi phí
│   ├── m5_report.py             # Mission 5: Báo cáo tổng hợp
│   ├── m6_your_turn.py          # (Optional) đo lường 5 phần mở rộng "Your Turn"
│   └── run_all.py               # Chạy M1–M5 liên tiếp
├── tests/                    # 15 unit + integration tests (pytest)
├── outputs/                  # Kết quả sinh ra (report.md, savings.png, ...)
├── bonus/                    # Phần mở rộng không bắt buộc
│   ├── litellm_tracker/      # Giả lập LiteLLM proxy + budget cap
│   ├── local_model/          # Đo tok/s với model chạy CPU
│   └── docker/               # Prometheus + Grafana dashboard
├── verify.py                 # Kiểm tra toàn bộ (11 checks)
└── requirements.txt
```

---

## Các module `finops/`

| Module | Chức năng |
|---|---|
| `metrics.py` | `compute_mfu`, `compute_mbu`, `roofline_regime`, `flag_util_lies`, `idle_waste_usd` |
| `pricing.py` | `request_cost`, `dollars_per_million`, `discount_stack`, `break_even_utilization`, `recommend_tier`, `spot_checkpoint_cost` |
| `allocation.py` | `cost_by_tag`, `tag_coverage`, `chargeback_ready`, `to_focus_rows` |
| `sustainability.py` | `wh_per_query`, `carbon_g`, `energy_cost_usd`, `tokens_per_watt` |
| `report.py` | `build_report`, `savings_waterfall` |

---

## 5 Missions

| # | Tên | Bạn học được gì | Chạy lệnh |
|---|---|---|---|
| **M1** | **Kiểm toán hiệu quả** — MFU/MBU, "GPU-Util lie", lãng phí idle | Slide §5 | `python missions/m1_efficiency_audit.py` |
| **M2** | **Đòn bẩy chi phí Inference** — `$/1M-token`, batch × cache × cascade | Slide §7 | `python missions/m2_inference_levers.py` |
| **M3** | **Chiến lược mua GPU** — điểm hòa vốn, spot/reserved, checkpoint sim | Slide §4 | `python missions/m3_purchasing.py` |
| **M4** | **Phân bổ chi phí** — tag → showback → chargeback, FOCUS export | Slide §10 | `python missions/m4_allocation.py` |
| **M5** | **Báo cáo tối ưu** — baseline vs. optimized + tính bền vững | Slide §1/§11 | `python missions/m5_report.py` |

**Bài học cốt lõi của M1:** GPU `gpu-h100-4` hiển thị **98% GPU-Util nhưng MFU chỉ ~0.20** — bạn trả tiền cho cả giờ H100 nhưng chỉ nhận được 1/5 FLOPs. `nvidia-smi` đo "clock đang bận", không đo hiệu quả tính toán.

---

## Dữ liệu đầu vào (`data/`)

| File | Số dòng | Cột quan trọng |
|---|---|---|
| `price_catalog.csv` | 7 GPU | `on_demand_hr`, `spot_hr`, `reserved_3yr_hr`, `peak_tflops_fp16`, `peak_bw_tbs`, `watts` |
| `gpu_telemetry.csv` | 11 GPU × 24 giờ | `gpu_util_pct`, `achieved_tflops`, `achieved_bw_tbs`, `power_w` |
| `token_usage.csv` | 2,400 request | `route_tier`, `input/output/cached_input_tokens`, `is_batch`, `is_reasoning`, `team`, `project` |
| `workloads.csv` | 8 job | `hours_per_day`, `days`, `interruptible`, `gpu_type`, `num_gpus` |

> `data/generate.py` dùng seed cố định — tái tạo luôn cho kết quả giống nhau.

---

## Phần mở rộng "Your Turn" (nơi học sâu nhất)

Lab **vẫn pass** mà không cần làm các phần này. Cả 5 phần đã được triển khai đầy đủ trong
`missions/m6_your_turn.py` (đo lường thực tế, không chỉ mô tả) — chạy độc lập, **không** đụng
vào M1–M5/`verify.py` đã được chấm điểm:

```bash
python missions/m6_your_turn.py    # -> outputs/your_turn.md
```

1. **Cải thiện chính sách tier.** `pricing.recommend_tier_v2()` (finops/pricing.py) tính thêm tỷ lệ
   gián đoạn (interrupt_rate + checkpoint rework) trước khi chọn spot, và so sánh break-even 1yr vs
   3yr reserved riêng biệt thay vì gộp chung "reserved". Kết quả đo: **1/8 job** đổi khuyến nghị
   (`job-infer-search`: reserved → on_demand, vì duty 75% không đủ an toàn cho cam kết dài hạn).
2. **Right-sizing theo MBU.** `metrics.right_size_for_memory_bound()` + `pricing.dollars_per_gb_vram()`
   dùng roofline (`arithmetic_intensity`/`roofline_regime`) để lọc các GPU inference/embedding bị
   memory-bound, rồi tìm phương án gộp (consolidate) sang GPU có `$/GB-VRAM` rẻ hơn. Kết quả đo:
   gộp 2×A100 + 2×A10G → tiết kiệm **~$1,210/tháng**.
3. **Kinh tế học của Cache.** `pricing.cache_is_worth_it()` / `cache_breakeven_hit_frac()` tính
   ngưỡng hoà vốn (~27.8%) giữa chi phí ghi cache một lần và mức giảm giá khi đọc lại. Áp dụng theo
   từng segment (team/project): **~50% chi phí inference/ngày** nằm ở các segment (`search`, `eval`)
   có tỷ lệ tái sử dụng cache quá thấp (~7%) — nên tắt cache ở đó.
4. **Ngân sách Reasoning.** `sustainability.wh_per_query(is_reasoning=True)` kết hợp `pricing.request_cost`
   để đo `$` và `Wh` của traffic `is_reasoning`: chỉ **8.4%** số request nhưng chiếm **94%** năng
   lượng. Đề xuất quy tắc: giới hạn reasoning ở 5% request, phần vượt route sang phản hồi thường
   → tiết kiệm **~$9/tháng + ~358 kWh/tháng**.
5. **Lịch trình nhận thức Carbon.** `sustainability.carbon_aware_region_rank()` /
   `region_shift_savings()` xếp hạng vùng theo điểm số kết hợp giá điện + carbon, rồi tính mức tiết
   kiệm nếu chuyển các job training gián đoạn được sang vùng đó. Kết quả đo: chuyển sang
   `us-east-wa` → tiết kiệm **~$116 + ~519 kg CO2e**.

Xem chi tiết đầy đủ (bảng số liệu từng job/segment) trong `outputs/your_turn.md` sau khi chạy, và
test tương ứng trong `tests/test_your_turn.py`.

---

## Đầu ra & Chấm điểm

Nộp `outputs/report.md` + `savings.png` + một bài viết ngắn:

| Thành phần | Trọng số |
|---|---|
| `verify.py` pass 11/11 | 30% |
| `pytest` pass | 20% |
| Báo cáo thể hiện baseline vs. optimized theo **$/1M-token** với từng lever | 30% |
| ≥2 phần mở rộng "Your Turn" được thực hiện và đo lường | 20% |

> Đây là **đầu vào Milestone 2** — mang báo cáo đến buổi demo platform.

---

## Bonus (tùy chọn, không chấm điểm)

- **`bonus/litellm_tracker/`** — proxy LiteLLM giả lập với budget cap theo API key
- **`bonus/local_model/`** — đo tok/s thực tế trên CPU, so sánh với simulation
- **`bonus/docker/`** — Prometheus + Grafana với dashboard chi phí GPU tích hợp sẵn

---

## Lưu ý quan trọng

- **Giá thay đổi hàng tháng.** Tất cả số liệu là snapshot tháng 6/2026. Cập nhật lại trước khi áp dụng thực tế.
- Bài học cốt lõi **bền vững theo thời gian**: đo `$/token`, dùng MFU thay vì GPU-Util, chồng chiết khấu, tính điểm hòa vốn trước khi commit reserved, gắn tag trước khi chargeback.
- Path graded thuần Python: `pandas`, `matplotlib`, `pytest`. Không cần GPU / API key / Docker.