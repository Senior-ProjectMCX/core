# DCS จำลองการไหลของเลือดสองชั้น — สรุปขั้นตอนการทำงาน

## ภาพรวม
จำลองและตรวจสอบความถูกต้องของเส้นโค้ง autocorrelation **g₂(τ)** สำหรับ Diffuse Correlation Spectroscopy (DCS) บน phantom เนื้อเยื่อสองชั้น จากนั้นเปรียบเทียบกับข้อมูลจริงจากการทดลอง (1540 trials, 3 ระยะ SDS)

---

## Pipeline (3 ขั้นตอน)

### ขั้นที่ 1 — `simulation.py` · MCX Monte Carlo
**ทำอะไร:** สร้างตาราง voxel 3 มิติของ phantom แล้วยิงโฟตอน 100 ล้านตัวผ่าน GPU (`pmcx`) บันทึก **partial pathlengths** (Lₛ, L_d) ของโฟตอนที่ตรวจจับได้ในแต่ละชั้น

**ทำไมถึงใช้วิธีนี้:** Monte Carlo เป็น gold standard สำหรับการจำลองการเดินทางของโฟตอนในเนื้อเยื่อที่มีการกระเจิงแสง GPU (MCX) ทำให้ 100 ล้านโฟตอนรันได้จริงในเวลาอันสั้น Partial pathlengths คือค่าสถิติขั้นต่ำที่จำเป็นสำหรับ DCS โดยไม่ต้องเก็บ trajectory ทั้งหมด

```
Input : ประเภท geometry (slab / realistic), จำนวนโฟตอน
Output: pathlengths.npz  →  {det_1_Ls, det_1_Ld, det_2_Ls, …}
```

---

### ขั้นที่ 2 — `dynamic_flow.py` + `generate_dataset.py` · สังเคราะห์ g₂
**ทำอะไร:**
1. แปลงอัตราการไหล (ml/min) → สัมประสิทธิ์การแพร่ Brownian D_B โดยใช้ **scaling factors ที่ปรับเทียบแล้ว** (scale_s = 0.0962, scale_d = 0.5492)
2. คำนวณ g₁(τ) = ⟨exp(−Γᵢ τ)⟩ โดยเฉลี่ยจากโฟตอนทั้งหมดที่ตรวจจับได้
3. ใช้สมการ Siegert: **g₂(τ) = 1 + β |g₁(τ)|²**
4. วนลูปทั้ง 1540 trials แล้วบันทึกเป็นไฟล์ `.mat`

**ทำไมถึงใช้วิธีนี้:**
- การเฉลี่ยผ่านโฟตอนแต่ละตัวใช้การกระจายตัว pathlength จริงจาก Monte Carlo — ทำให้ถูกต้องทางฟิสิกส์โดยไม่ต้องใช้สูตรประมาณ (slab approximation)
- β (coherence factor) หาค่าจากข้อมูลจริงที่ τ→0 (g₂(0) − 1) เพื่อดูดซับผลจากการจัดวางออปติกจริง
- การลด sampling เหลือ 10k โฟตอน/detector ช่วยให้รันได้เร็วโดยมีความคลาดเคลื่อนเชิงสถิติน้อยมาก

```
Input : pathlengths.npz, Dataset_phantom-intra40.mat (อัตราไหล + ปรับ β)
Output: Dataset_simulated-intra40.mat
```

---

### ขั้นที่ 3 — `compare.py` · ตรวจสอบเชิงปริมาณ
**ทำอะไร:** โหลดทั้งข้อมูลจริงและข้อมูลจำลอง คำนวณ **MAE และ RMSE** ตลอด 1540 × 501 จุดของเส้นโค้งในแต่ละ SDS และพล็อตกราฟเปรียบเทียบ 4 ชุดการไหลตัวแทน

**ทำไมถึงใช้วิธีนี้:** MAE และ RMSE ร่วมกันวัดทั้งค่าเฉลี่ยและความเบี่ยงเบนขนาดใหญ่ ทำให้ประเมินคุณภาพของการจำลองได้รอบด้าน

```
Input : Dataset_phantom-intra40.mat, Dataset_simulated-intra40.mat
Output: g2_comparison.png + ค่า metrics ที่พิมพ์ออกมา
```

---

## การตัดสินใจออกแบบสำคัญ

| การตัดสินใจ | เหตุผล |
|---|---|
| ใช้ geometry แบบ **slab** (ไม่ใช่ realistic) | โมเดล realistic ทำให้แผ่น Mylar 0.05 mm กลายเป็น 0.5 mm (ขีดจำกัด voxel) สร้างชั้นกระเจิงหนาเกินจริง |
| MCX บน GPU ด้วย 100M โฟตอน | ได้การกระจายตัว pathlength ที่เสถียรทางสถิติ; รันไม่กี่นาทีบน RTX 4050 |
| ปรับเทียบ D_B scaling ด้วย Nelder-Mead | เชื่อมช่องว่างระหว่างอัตราไหลเชิงปริมาตรกับการแพร่ Brownian จริงใน intralipid |
| ปรับ β แยกตาม SDS จากข้อมูลจริง | คำนึงถึงการสูญเสีย spatial coherence และออปติกของ detector ในแต่ละระยะ |

---

## คำสั่งรันเร็ว

```powershell
# เปิด environment
.\venv_mcx\Scripts\Activate.ps1

# ขั้นที่ 1 - จำลอง Monte Carlo
python .\code\simulation.py --nphoton 1e8 --geometry simplified --output .\data\pathlengths.npz

# ขั้นที่ 2 - สร้าง dataset จำลอง
python .\code\generate_dataset.py --pathlengths .\data\pathlengths.npz --physical-mat .\data\Dataset_phantom-intra40.mat --output .\data\Dataset_simulated-intra40.mat

# ขั้นที่ 3 - เปรียบเทียบและพล็อต
python .\code\compare.py --physical .\data\Dataset_phantom-intra40.mat --simulated .\data\Dataset_simulated-intra40.mat --output-plot .\plots\g2_comparison.png
```
