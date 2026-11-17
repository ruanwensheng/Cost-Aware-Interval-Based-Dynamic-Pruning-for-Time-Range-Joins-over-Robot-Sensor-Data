# EDA bag 5: chỉ metadata, không giải nén

```
(E:\ros-lab\.conda) PS E:\ros-lab>
```

Mọi lệnh dưới đây **không giải nén chunk nào**. Chúng chỉ đọc bag header, header của các chunk, record INDEX DATA và phần index ở cuối file. Mỗi lệnh lưu output vào `results\eda\*.txt` để đưa vào báo cáo.

```powershell
$bag = "data\bags\2023-12-26-15-10-15.compressed.bag"
```

---

## 1. Cấu trúc chung của file bag

```powershell
# Bag header + tổng quan: cỡ file, index_pos, số connection, số chunk, kiểu nén, thời lượng
python eda_bag.py --bag 5 --action info | Tee-Object results\eda\info.txt

# Thứ tự các record ở đầu file: magic line, BAG HEADER, CHUNK, INDEX DATA, CHUNK, ...
python tools\inspect_bag.py $bag --limit 15 | Tee-Object results\eda\layout_head.txt

# Phần index ở cuối file: BAG HEADER rồi nhảy tới index_pos: CONNECTION x 38, rồi CHUNK INFO
python tools\inspect_bag.py $bag --index --limit 42 | Tee-Object results\eda\layout_index.txt
```

---

## 2. Metadata từng loại record, mỗi loại 1–2 bản ghi mẫu

```powershell
# BAG HEADER (op 0x03)
python tools\inspect_bag.py $bag --limit 1

# CHUNK (op 0x05): kiểu nén, cỡ sau giải nén, cỡ nén
python tools\inspect_bag.py $bag --limit 30 | Select-String 'CHUNK ' | Select-Object -First 2

# INDEX DATA (op 0x04): conn, số message, 3 entry (time, offset trong chunk) đầu tiên
python tools\inspect_bag.py $bag --limit 30 | Select-String 'INDEX DATA' | Select-Object -First 2

# CONNECTION (op 0x07): conn id, topic, kiểu message
python tools\inspect_bag.py $bag --index --limit 42 | Select-String 'CONNECTION' | Select-Object -First 2

# CHUNK INFO (op 0x06): chunk_pos, start, end, (conn, số message) cho mỗi connection
python tools\inspect_bag.py $bag --index --limit 42 | Select-String 'CHUNK INFO' | Select-Object -First 2

# CONNECTION của các topic quan tâm
python tools\inspect_bag.py $bag --index --limit 42 | Select-String 'topic=/realsense/imu |topic=/reach_1/fix |topic=/realsense/color/image_raw |topic=/odom '
```

MESSAGE DATA (op 0x02) nằm **bên trong** chunk đã nén, nên bước này không xem được. Metadata của nó (thời gian và vị trí của từng message) đã có trong INDEX DATA và CHUNK INFO.

---

## 3. Topic và message type

```powershell
# Mọi topic: kiểu, số message, Hz  ->  results\eda\topics.csv
python eda_bag.py --bag 5 --action info | Select-String '^/' | Tee-Object results\eda\topics.txt

# Số topic theo từng message type
Import-Csv results\eda\topics.csv | Group-Object type | Sort-Object Count -Descending | Format-Table Name, Count

# Kiểm tra config\topics.yaml (4 role phải OK)
python eda_bag.py --bag 5 --action info | Select-String 'OK$|NOT IN BAG'

# Mỗi topic xuất hiện ở bao nhiêu chunk / tổng số chunk   (cần chunks.csv: chạy mục 4 trước)
python -c "import pandas as pd; d=pd.read_csv('results/eda/chunks.csv'); t=d.columns[9:]; r=pd.DataFrame({'chunks':(d[t]>0).sum(),'pct_chunks':((d[t]>0).mean()*100).round(2),'messages':d[t].sum(),'max_msgs_per_chunk':d[t].max()}); r.to_csv('results/eda/topic_chunks.csv'); print(r.to_string()); print('total chunks', len(d))" | Tee-Object results\eda\topic_chunks.txt

# Chỉ các topic quan tâm
python -c "import pandas as pd; r=pd.read_csv('results/eda/topic_chunks.csv',index_col=0); print(r.loc[['/realsense/imu','/reach_1/fix','/realsense/color/image_raw','/odom']].to_string())"

# Chunk chứa nhiều role cùng lúc; L1 phải đọc bao nhiêu % chunk / byte
python eda_bag.py --bag 5 --action topic-mix | Tee-Object results\eda\topic_mix.txt

# Tần số và chu kỳ thật trên cả bag (từ INDEX DATA; --limit 0 = không giải nén chunk nào)
python eda_bag.py --bag 5 --action rates --window 0 797 --limit 0 | Tee-Object results\eda\rates.txt
python eda_bag.py --bag 5 --action rates --window 0 797 --limit 0 | Select-String '/realsense/imu |/reach_1/fix|/realsense/color/image_raw|/odom |suggested'
```

Ở lệnh `rates --limit 0`, các cột `size_mean`, `size_max`, `mb_per_s` sẽ là `NaN`. Đúng như mong đợi, vì muốn đo cỡ message phải giải nén.

---

## 4. Chunk

```powershell
# Mỗi chunk một dòng  ->  results\eda\chunks.csv ; tổng kết in ra màn hình
python eda_bag.py --bag 5 --action chunks --limit 5 | Tee-Object results\eda\chunks.txt

# 2 bản ghi mẫu (đủ mọi cột, kể cả số message mỗi topic)
Import-Csv results\eda\chunks.csv | Select-Object -First 2 | Format-List

# Phân bố: byte nén, byte giải nén, tỉ lệ nén, số ms dữ liệu, số message, số topic mỗi chunk
python -c "import pandas as pd; d=pd.read_csv('results/eda/chunks.csv'); d['dur_ms']=(d.end_s-d.start_s)*1000; d['ratio']=d.bytes_uncompressed/d.bytes_compressed; print(d[['bytes_compressed','bytes_uncompressed','ratio','dur_ms','n_messages','n_topics']].describe(percentiles=[.05,.5,.95]).round(2).to_string())" | Tee-Object results\eda\chunk_stats.txt

# Kiểu nén
Import-Csv results\eda\chunks.csv | Group-Object compression | Format-Table Name, Count

# Số topic mỗi chunk: bao nhiêu chunk có 1, 2, ... topic
Import-Csv results\eda\chunks.csv | Group-Object n_topics | Sort-Object { [int]$_.Name } | Format-Table Name, Count

# Thời gian các chunk có chồng lên nhau không (ảnh hưởng tới việc bỏ chunk theo thời gian, L2)
python -c "import pandas as pd; d=pd.read_csv('results/eda/chunks.csv'); print('chunks:', len(d), ' start không tăng dần:', int((d.start_s.diff()<0).sum()), ' chồng lên chunk trước:', int((d.start_s.values[1:]<d.end_s.values[:-1]).sum()))"

# Cửa sổ của ma trận chạy (300..420 s): số chunk và số GB nén phải đọc ở mức L2
python -c "import pandas as pd; d=pd.read_csv('results/eda/chunks.csv'); w=d[(d.end_s>=300)&(d.start_s<=420)]; print(len(w), 'chunks /', len(d), ' ', round(w.bytes_compressed.sum()/2**30,2), 'GB /', round(d.bytes_compressed.sum()/2**30,2), 'GB')"
```

---

## Thứ tự chạy

1. Mục 1
2. Mục 4, dòng đầu tiên (tạo `chunks.csv`)
3. Mục 2, mục 3, phần còn lại của mục 4

Các file nộp kèm: `results\eda\*.txt`, `topics.csv`, `chunks.csv`, `topic_chunks.csv`, `rates.csv`.
