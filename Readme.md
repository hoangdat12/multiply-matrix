# Intel Intrinsics - AVX

**Intrinsics** là các hàm tích hợp sẵn được cung cấp bởi trình biên dịch trong C/C++, cho phép lập trình viên gọi trực tiếp các tập lệnh hợp ngữ (assembly) cấp thấp mà không cần viết mã assembly thủ công. 

## Quy chuẩn đặt tên

`_mm<độ_rộng_bit>_<tên_thao_tác>_<kiểu_dữ_liệu>`

---

### 1. Kích thước thanh ghi

*   `_mm_`: Thanh ghi **128-bit** (SSE).
*   `_mm256_`: Thanh ghi **256-bit** (AVX/AVX2).
*   `_mm512_`: Thanh ghi **512-bit** (AVX-512).

### 2. Tên thao tác

*   `add`, `sub`, `mul`, `div`: Phép toán số học cơ bản (Cộng, trừ, nhân, chia).
*   `fmadd`: Fused multiply-add (Nhân rồi cộng biểu thức $A \times B + C$).
*   `load`, `store`: Tương tác bộ nhớ (Đọc từ bộ nhớ vào thanh ghi hoặc ghi từ thanh ghi ra bộ nhớ).
*   `set1`, `setzero`: Khởi tạo giá trị cho vector.

### 3. Kiểu dữ liệu 

*   `ps` (Packed Single): Số thực 32-bit (float). *Ví dụ: thanh ghi 256-bit chứa 8 số float.*
*   `pd` (Packed Double): Số thực 64-bit (double).
*   `epi8`, `epi16`, `epi32`, `epi64`: Số nguyên có dấu (Extended Packed Integer) 8, 16, 32, hoặc 64-bit.
*   `epu8`, `epu16`, `epu32`: Số nguyên không dấu (Unsigned).
*   `si128`, `si256`: Signed Int.

## 2. Intrinsics Life Cycle

* Bước 1: Load / Initialize 
* Bước 2: Compute / Process
* Bước 3: Store / Extract

## 3. Memory Alignment

Trong C, RAM bố trí như một chuỗi bit liên tục, còn CPU thì không. Nó đọc RAM theo từng khối cố định, thường được quyết định bởi kiến trúc bus và cache (ví dụ: các khối 32 byte hoặc 64 byte).

Với thanh ghi AVX (256-bit = 32 bytes), CPU thích nhất là được lấy một khối dữ liệu có địa chỉ bắt đầu là bội số của 32 (0, 32, 64, 96...). Đây là Memory Alignment.