# EDA bag 5: kết quả metadata (không giải nén)

File: `data/bags/2023-12-26-15-10-15.compressed.bag`. Output gốc nằm trong `results/eda/`:
`info.txt`, `layout_head.txt`, `layout_index.txt`, `chunks.txt`, `chunk_stats.txt`, `topic_chunks.txt`, `topic_mix.txt`, `rates.txt`, `topics.csv`, `chunks.csv`, `topic_chunks.csv`, `rates.csv`.

Không lệnh nào giải nén chunk.

---

## 1. Cấu trúc file

| Mục | Giá trị |
|---|---|
| Cỡ file | 31 525 138 787 B (29.4 GiB = 31.5 GB) |
| Định dạng | `#ROSBAG V2.0` (13 B đầu file) |
| Thời lượng | 796.5 s (Unix 1703614215.727 → 1703615012.181) |
| Số connection | 38 (38 topic, mỗi topic 1 connection) |
| Số chunk | 46 809, tất cả nén **bz2** |
| Phần index | bắt đầu ở `index_pos = 31 516 148 735`, cỡ 8.6 MB |

Bố cục:

```
@0            magic "#ROSBAG V2.0\n"
@13           BAG HEADER            (data 4027 B, phần đệm)
@4117         CHUNK 0               (data 1 616 264 B nén bz2)
@1620429      INDEX DATA x 29       (một record cho mỗi connection có mặt trong chunk 0)
@1627040      CHUNK 1
...           CHUNK + INDEX DATA    x 46 809
@31516148735  CONNECTION x 38       (phần index)
@31516245359  CHUNK INFO x 46 809
EOF
```

---

## 2. Bản ghi mẫu theo từng loại record

**BAG HEADER (op 0x03)**
```
@13  BAG HEADER  data=4027 bytes  index_pos=31516148735 conn_count=38 chunk_count=46809
```

**CHUNK (op 0x05)**
```
@4117     CHUNK  data=1616264 bytes  compression=bz2 uncompressed_size=2905760
@1627040  CHUNK  data=248156 bytes   compression=bz2 uncompressed_size=1882474
```

**INDEX DATA (op 0x04)**: mỗi entry là (thời gian, offset trong chunk đã giải nén); chỉ in 3 entry đầu.
```
@1620429  INDEX DATA  data=468 bytes  conn=0 count=39  [('1703614215.726730591', 2734), ('1703614215.729920857', 14770), ('1703614215.735016353', 16575)]
@1620952  INDEX DATA  data=456 bytes  conn=1 count=38  [('1703614215.727247005', 5579), ('1703614215.729363576', 11956), ('1703614215.735103174', 17515)]
```

**CONNECTION (op 0x07)**
```
@31516148735  CONNECTION  data=2683 bytes  conn=0 topic=/reach_2/imu     type=sensor_msgs/Imu
@31516151469  CONNECTION  data=2421 bytes  conn=1 topic=/reach_1/imu/mag type=sensor_msgs/MagneticField
```
Connection của các topic quan tâm:
```
conn=12  /reach_1/fix                sensor_msgs/NavSatFix
conn=26  /odom                       nav_msgs/Odometry
conn=28  /realsense/color/image_raw  sensor_msgs/Image
conn=31  /realsense/imu              sensor_msgs/Imu
```

**CHUNK INFO (op 0x06)**
```
@31516245359  CHUNK INFO  chunk_pos=4117    start=1703614215.726730591 end=1703614215.892027051 connections=29
              (conn, messages)=[(0,39), (1,38), (2,38), (3,38), (4,38), (5,37), (6,37), (7,31), (8,27), (9,27), (10,25), (11,26), (12,1), ... (28,1)]
@31516245699  CHUNK INFO  chunk_pos=1627040 start=1703614215.892064333 end=1703614215.903751764 connections=18
              (conn, messages)=[(5,3), (2,3), (1,3), (4,3), (3,3), (7,3), (10,3), (6,3), (29,1), (30,1), (9,2), (31,3), (11,2), (8,2), (0,2), (32,1), (33,1), (34,1)]
```

**MESSAGE DATA (op 0x02)**: không xem được, vì nằm bên trong chunk nén. Thời gian và offset của từng message đã có trong INDEX DATA.

**Một dòng mẫu trong `chunks.csv`** (chunk 0, rút gọn):
```
chunk=0  chunk_pos=4117  bz2  bytes_compressed=1616264  bytes_uncompressed=2905760
start_s=0.000  end_s=0.165  n_messages=418  n_topics=29
/realsense/imu=0  /reach_1/fix=1  /realsense/color/image_raw=1  /odom=1  /reach_1/imu=38 ...
```

---

## 3. Topic và message type

38 topic thuộc 13 message type:

| Type | Số topic |
|---|---|
| `sensor_msgs/TimeReference` | 9 |
| `sensor_msgs/Imu` | 4 |
| `sensor_msgs/CameraInfo` | 4 |
| `sensor_msgs/Image` | 4 |
| `geometry_msgs/TwistStamped` | 3 |
| `sensor_msgs/MagneticField` | 3 |
| `sensor_msgs/NavSatFix` | 3 |
| `realsense2_camera/IMUInfo` | 2 |
| `tf2_msgs/TFMessage` | 2 |
| `std_msgs/Float32MultiArray`, `wheel_odometry/Distances`, `nav_msgs/Odometry`, `wheel_control/WheelControl` | 1 mỗi loại |

Theo nhóm:

| Nhóm | Topic | Hz | Số chunk có mặt / 46 809 |
|---|---|---|---|
| IMU của Reach (×3) | `/reach_{1,2,3}/imu`, `/imu/mag`, `/imu/driver_t`, `/imu/system_t` | 200–230 | 52–55% |
| GNSS của Reach (×3) | `/reach_{1,2,3}/fix`, `/vel`, `/time_reference` | 5 | 8.51% |
| Camera RealSense | 4 luồng ảnh + 4 `camera_info` | 14.6–15 | 25% mỗi luồng |
| IMU của RealSense | `/realsense/imu` | 199 | 55.4% |
| Bánh xe | `/odom`, `/distance`, `/nav_cmd` | 10 | 17.0% |
| Khác | `/surcos_status` (0.33 Hz); `/tf`, `/tf_static`, `*/imu_info` (1 message) | — | ≤ 0.6% |

### Topic quan tâm

| Role | Topic | Messages | Hz | Chu kỳ trung vị | Chu kỳ p95 | Khoảng trống lớn nhất | Chunk có mặt | % chunk | Tối đa msg/chunk |
|---|---|---|---|---|---|---|---|---|---|
| imu | `/realsense/imu` | 158 462 | 198.96 | 4.98 ms | 9.88 ms | 43.8 ms | 25 948 | 55.43% | 18 |
| gnss | `/reach_1/fix` | 3 982 | 5.00 | 0.200 s | 0.206 s | 0.223 s | 3 982 | 8.51% | 1 |
| image | `/realsense/color/image_raw` | 11 933 | 14.98 | 66.7 ms | 73.4 ms | 133 ms | 11 933 | 25.49% | 1 |
| odom | `/odom` | 7 963 | 10.00 | 0.100 s | 0.104 s | 0.122 s | 7 963 | 17.01% | 1 |

So sánh 3 ăng-ten GNSS (khoảng trống lớn nhất trên cả bag): `/reach_1/fix` 0.223 s, `/reach_3/fix` 0.327 s, `/reach_2/fix` 0.368 s. `/reach_1/fix` đều nhất, khớp với lựa chọn trong `config/topics.yaml`. Giá trị `--tol` gợi ý: **0.100 s** (nửa chu kỳ GNSS).

### Topic cùng nằm trong chunk (topic-mix)

| Chunk chứa | Số chunk | % chunk | % byte |
|---|---|---|---|
| imu | 25 948 | 55.4% | 74.7% |
| gnss | 3 982 | 8.5% | 17.7% |
| image | 11 933 | 25.5% | 58.9% |
| odom | 7 963 | 17.0% | 35.9% |
| image AND imu | 11 840 | 25.3% | 58.4% |
| gnss AND imu | 3 790 | 8.1% | 17.5% |
| image AND imu AND gnss | 3 475 | 7.4% | 17.1% |

L1 (bỏ chunk không chứa topic cần) phải đọc:

| Topic cần | % chunk | % byte |
|---|---|---|
| imu | 55.4% | 74.7% |
| gnss | 8.5% | 17.7% |
| imu + gnss | 55.8% | 74.9% |

---

## 4. Chunk

KB, MB ở bảng dưới tính theo cơ số 1024.

| Đại lượng | min | 5% | trung vị | trung bình | 95% | max |
|---|---|---|---|---|---|---|
| Byte nén | 4.3 KB | 223 KB | 433 KB | 656 KB | 1.55 MB | 1.66 MB |
| Byte sau giải nén | 22.7 KB | 900 KB | 1.76 MB | 1.55 MB | 2.67 MB | 2.77 MB |
| Tỉ lệ nén | 1.61 | 1.73 | 2.08 | 3.44 | 8.07 | 8.85 |
| Khoảng thời gian (ms) | 0 | 0 | 1.46 | 16.6 | 62.9 | 165.3 |
| Số message | 1 | 1 | 10 | 51.5 | 185 | 418 |
| Số topic | 1 | 1 | 9 | 10.3 | 25 | 30 |

- Tổng: 29.3 GB nén → 71.1 GB sau giải nén (2.43×).
- Thứ tự thời gian: start tăng dần theo vị trí trong file; **0** chunk chồng thời gian lên chunk trước.
- Số topic mỗi chunk: 3 914 chunk chỉ có 1 topic, 4 488 chunk có 2 topic, 5 chunk có 30 topic. Phân bố đầy đủ trong `topic_mix.txt`.
- Cửa sổ 300–420 s: **7 076 / 46 809 chunk, 4.26 / 29.3 GB** (14.5% số byte). Đây là phần L2 phải đọc.

---

## 5. Nhận xét

1. **GNSS chỉ nằm trong 8.5% số chunk (17.7% số byte).** Điều này khác với giả định ở CLAUDE.md mục 2 và 4 ("mỗi chunk trộn mọi topic", "đọc GNSS ... phải giải nén gần hết các chunk"). Ở pha 1 của L3/L4 trên bag, nếu chỉ đọc các chunk có GNSS trong cửa sổ thì phải giải nén khoảng 1/6 số byte, không phải gần hết.
2. **IMU nằm trong 55% số chunk nhưng chiếm 75% số byte**, vì các chunk có IMU cũng thường có ảnh (image AND imu = 58.4% số byte). L1 với query chỉ cần IMU tiết kiệm được khoảng 25% số byte.
3. **Chunk rất ngắn về thời gian** (trung vị 1.5 ms, max 165 ms) và không chồng nhau. Vì vậy bỏ chunk theo thời gian (L2) cắt rất sát cửa sổ.
4. **Có nhiều chunk chỉ chứa 1 ảnh** (ví dụ chunk 2 và 3: 1 message, khoảng 924 KB sau giải nén).
5. **Thời gian đọc metadata phụ thuộc mạnh vào cache của hệ điều hành.** `chunks` đọc header của 46 809 chunk mất 584 s (khoảng 12 ms mỗi chunk, giới hạn bởi seek). Ngay sau đó, `rates` đọc INDEX DATA của cùng các chunk chỉ mất 16 s. Cần nhớ điều này khi đo với `--cache cold`/`warm`.
