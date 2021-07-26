## Results

### RN32 on CIFAR10
| backend | quant mode | #GPUs/node | #nodes | #BITS | #Epochs | Train Acc | Train time  | Test Acc | batch\_size | bucket\_size | speedup             |
| ------- | ---------- | ---------- | ------ | ----- | ------- | --------- | ----------- | -------- | ----------- | ------------ | ------------------- |
| NCCL    | \-         | 8          | 1      | 32    | 300     | 99.952    | 1125.284725 | 93.31    | 256         | \-           | \-                  |
| QNCCL   | stochastic | 8          | 1      | 8     | 300     | 99.96     | 1153.662566 | 93.53    | 256         | 1024         | 0.975401956795108 x |
| QNCCL   | stochastic | 8          | 1      | 4     | 300     | 99.9      | 1173.232386 | 92.84    | 256         | 1024         | 0.959132000281912 x |


### RN20 on CIFAR10

| backend | quant mode    | #GPUs/node | #nodes | #BITS | #Epochs | Train Acc | Train time  | Test Acc | batch\_size | bucket\_size | speedup             |
| ------- | ------------- | ---------- | ------ | ----- | ------- | --------- | ----------- | -------- | ----------- | ------------ | ------------------- |
| NCCL    | \-            | 4          | 1      | 32    | 300     | 99.802    | 1120.968    | 92.46    | 256         | \-           | \-                  |
| QNCCL   | deterministic | 4          | 1      | 8     | 300     | 99.79     | 1188.731    | 92.73    | 256         | 512          | 0.942995513703269 x |
| QNCCL   | deterministic | 4          | 1      | 4     | 300     | 99.729    | 1220.549    | 92.479   | 256         | 512          | 0.918412943683539 x |
| QNCCL   | deterministic | 4          | 1      | 2     | 300     | 89.054    | 1189.14     | 85.08    | 256         | 512          | 0.942671174125839 x |
| NCCL    | \-            | 8          | 1      | 32    | 300     | 99.852    | 767.7633129 | 92.59    | 256         | \-           | \-                  |
| QNCCL   | stochastic    | 8          | 1      | 8     | 300     | 99.866    | 798.3200989 | 92.8     | 256         | 1024         | 0.961723641919039 x |
| QNCCL   | stochastic    | 8          | 1      | 4     | 300     | 99.684    | 800.6983956 | 91.86    | 256         | 1024         | 0.958867055436563 x |


### RN50 on ImageNet
| backend | quant mode    | #GPUs/node | #nodes | #BITS | #Epochs | Train Acc   | Train time  | Test Acc | batch\_size | bucket\_size | speedup            |
| ------- | ------------- | ---------- | ------ | ----- | ------- | ----------- | ----------- | -------- | ----------- | ------------ | ------------------ |
| NCCL    | \-            | 8          | 1      | 32    | 90      | 78.08546576 | 122497.7266 | 76.058   | 32          | \-           | \-                 |
| QNCCL   | stochastic    | 8          | 1      | 8     | 90      | 78.1296442  | 108288.6616 | 76.254   | 32          | 1024         | 1.13121470642673 x |
| QNCCL   | stochastic    | 8          | 1      | 4     | 90      | 76.94658312 | 103929.3185 | 75.834   | 32          | 1024         | 1.17866381040035 x |
| QNCCL   | stochastic    | 8          | 1      | 4     | 90      | 77.4568987  | 107348.4664 | 75.886   | 32          | 128          | 1.14112227874107 x |
| QNCCL   | deterministic | 8          | 1      | 8     | 90      | 78.12004359 | 106903.067  | 76.248   | 32          | 1024         | 1.14587663370917 x |
| QNCCL   | deterministic | 8          | 1      | 4     | 90      | 73.43689508 | 103794.4613 | 74.294   | 32          | 1024         | 1.1801952151885 x  |
| QNCCL   | deterministic | 8          | 1      | 4     | 90      | 76.81365754 | 102327.0835 | 75.646   | 32          | 256          | 1.19711930033255 x |

### RN152 on CIFAR10

| backend | quant mode    | #GPUs/node | #nodes | #BITS | #Epochs | Train Acc | Train time  | Test Acc | batch\_size | bucket\_size | speedup            |
| ------- | ------------- | ---------- | ------ | ----- | ------- | --------- | ----------- | -------- | ----------- | ------------ | ------------------ |
| NCCL    | \-            | 8          | 1      | 32    | 10      | \-        | 163.2915887 | \-       | 256         | \-           | \-                 |
| QNCCL   | fake          | 8          | 1      | 8     | 10      | \-        | 147.6970984 | \-       | 256         | \-           | 1.10558427000203 x |
| QNCCL   | stochastic    | 8          | 1      | 8     | 10      | \-        | 151.0303982 | \-       | 256         | 1024         | 1.08118359372412 x |
| QNCCL   | deterministic | 8          | 1      | 8     | 10      | \-        | 151.1828918 | \-       | 256         | 1024         | 1.08009303687936 x |
| QNCCL   | deterministic | 8          | 1      | 4     | 10      | \-        | 150.2915841 | \-       | 256         | 1024         | 1.08649855376922 x |
| QNCCL   | deterministic | 8          | 1      | 2     | 10      | \-        | 150.3734502 | \-       | 256         | 512          | 1.08590704353243 x |



### BERT on Wikipedia

| backend | quant mode    | #GPUs/node | #nodes | GPU arch | #BITS | #Epochs | Train time  | loss        | training\_sequences\_per\_second | raw\_train\_time | batch\_size | bucket\_size | train type | speedup            |
| ------- | ------------- | ---------- | ------ | -------- | ----- | ------- | ----------- | ----------- | -------------------------------- | ---------------- | ----------- | ------------ | ---------- | ------------------ |
| NCCL    | \-            | 8          | 1      | RTX3090  | 32    | 1       | 1219.732954 | 11.29370403 | 13.6777045                       | 1197.86182       | 4           | \-           | scratch    | \-                 |
| QNCCL   | deterministic | 8          | 1      | RTX3090  | 8     | 1       | 742.7801163 | 11.29370403 | 22.74485021                      | 720.3388834      | 4           | 1024         | scratch    | 1.66291428569046 x |


### Transformer-XL-Base on WT-103

| backend | version | quant mode    | #GPUs/node | #nodes | GPU arch | #BITS | #Steps | Train time | loss | perplexity | token per second | batch\_size | bucket\_size | train type | speedup            |
| ------- | ------- | ------------- | ---------- | ------ | -------- | ----- | ------ | ---------- | ---- | ---------- | ---------------- | ----------- | ------------ | ---------- | ------------------ |
| NCCL    | 2.8.4   | \-            | 8          | 1      | RTX3090  | 32    | 200    | 283.8      | 8.12 | 3356.923   | 35780.6          | 256         | \-           | scratch    | \-                 |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 8     | 200    | 109.2      | 8.11 | 3340.454   | 98128.16         | 256         | 1024         | scratch    | 2.74249621303164 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 6     | 200    | 110.4      | 8.11 | 3340.537   | 97275.22         | 256         | 1024         | scratch    | 2.7186581555368 x  |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 5     | 200    | 109.8      | 8.13 | 3394.557   | 98223.46         | 256         | 1024         | scratch    | 2.74515966752933 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 4     | 200    | 78.6       | 8.12 | 3372.592   | 141324.07        | 256         | 1024         | scratch    | 3.9497400826146 x  |


### RN50 on ImageNet

| backend | version | quant mode    | #GPUs/node | #nodes | GPU arch | #BITS | #Steps | loss    | throughput(img/sec) | batch\_size | bucket\_size | train type | speedup            |
| ------- | ------- | ------------- | ---------- | ------ | -------- | ----- | ------ | ------- | ------------------- | ----------- | ------------ | ---------- | ------------------ |
| NCCL    | 2.8.4   | \-            | 8          | 1      | RTX3090  | 32    | 200    | 7.01977 | 5409.95             | 256         | \-           | scratch    | \-                 |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 8     | 200    | 7.00762 | 6968.95             | 256         | 1024         | scratch    | 1.28817271878668 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 6     | 200    | 7.01791 | 7002.68             | 256         | 1024         | scratch    | 1.29440752687178 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 5     | 200    | 7.02619 | 7029.57             | 256         | 1024         | scratch    | 1.29937799794823 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 4     | 200    | 7.02254 | 7774.48             | 256         | 1024         | scratch    | 1.43707058290742 x |


### RN18 on ImageNet:

| backend | version | quant mode    | #GPUs/node | #nodes | GPU arch | #BITS | #Steps | loss    | throughput(img/sec) | batch\_size | bucket\_size | train type | speedup            |
| ------- | ------- | ------------- | ---------- | ------ | -------- | ----- | ------ | ------- | ------------------- | ----------- | ------------ | ---------- | ------------------ |
| NCCL    | 2.8.4   | \-            | 8          | 1      | RTX3090  | 32    | 200    | 6.248   | 11544.97            | 256         | \-           | scratch    | \-                 |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 8     | 200    | 6.2428  | 15317.22            | 256         | 1024         | scratch    | 1.32674402791865 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 6     | 200    | 6.2377  | 15586.62            | 256         | 1024         | scratch    | 1.35007886551459 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 5     | 200    | 6.23708 | 18285.3             | 256         | 1024         | scratch    | 1.58383261281753 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 4     | 200    | 6.27085 | 15898.34            | 256         | 1024         | scratch    | 1.37707936876406 x |


### RN34 on ImageNet

| backend | version | quant mode    | #GPUs/node | #nodes | GPU arch | #BITS | #Steps | loss    | throughput(img/sec) | batch\_size | bucket\_size | train type | speedup            |
| ------- | ------- | ------------- | ---------- | ------ | -------- | ----- | ------ | ------- | ------------------- | ----------- | ------------ | ---------- | ------------------ |
| NCCL    | 2.8.4   | \-            | 8          | 1      | RTX3090  | 32    | 200    | 6.3554  | 7615.17             | 256         | \-           | scratch    | \-                 |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 8     | 200    | 6.38394 | 9810.76             | 256         | 1024         | scratch    | 1.28831792330309 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 6     | 200    | 6.39897 | 9415.46             | 256         | 1024         | scratch    | 1.23640837958969 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 5     | 200    | 6.39303 | 9850.5              | 256         | 1024         | scratch    | 1.29353645420916 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 4     | 200    | 6.4088  | 10410.78            | 256         | 1024         | scratch    | 1.36711064887586 x |


### RN101 on ImageNet

| backend | version | quant mode    | #GPUs/node | #nodes | GPU arch | #BITS | #Steps | loss    | throughput(img/sec) | batch\_size | bucket\_size | train type | speedup            |
| ------- | ------- | ------------- | ---------- | ------ | -------- | ----- | ------ | ------- | ------------------- | ----------- | ------------ | ---------- | ------------------ |
| NCCL    | 2.8.4   | \-            | 8          | 1      | RTX3090  | 32    | 200    | 7.02027 | 3267.06             | 256         | \-           | scratch    | \-                 |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 8     | 200    | 7.01387 | 4120.82             | 256         | 1024         | scratch    | 1.26132363654172 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 6     | 200    | 7.02151 | 3927.55             | 256         | 1024         | scratch    | 1.20216647383274 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 5     | 200    | 7.01281 | 4154.95             | 256         | 1024         | scratch    | 1.27177033785728 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 4     | 200    | 7.03821 | 4623.64             | 256         | 1024         | scratch    | 1.41522959480389 x |


### Vision Transformer Base on ImageNet

| backend | version | quant mode    | #GPUs/node | #nodes | GPU arch | #BITS | #Steps | loss     | throughput(img/sec) | batch\_size | bucket\_size | train type | speedup            |
| ------- | ------- | ------------- | ---------- | ------ | -------- | ----- | ------ | -------- | ------------------- | ----------- | ------------ | ---------- | ------------------ |
| NCCL    | 2.7.8   | \-            | 8          | 1      | RTX3090  | 32    | 200    | 7.001426 | 645.26              | 64          | \-           | scratch    | \-                 |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 8     | 200    | 7.001441 | 1277.94             | 64          | 1024         | scratch    | 1.98050398289062 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 6     | 200    | 7.001479 | 1275.45             | 64          | 1024         | scratch    | 1.97664507330378 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 5     | 200    | 7.001877 | 1282.07             | 64          | 1024         | scratch    | 1.98690450361095 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 4     | 200    | 7.00335  | 1534.24             | 64          | 1024         | scratch    | 2.37770821064377 x |


### Vision Transformer small on ImageNet

| backend | version | quant mode    | #GPUs/node | #nodes | GPU arch | #BITS | #Steps | loss     | throughput(img/sec) | batch\_size | bucket\_size | train type | speedup            |
| ------- | ------- | ------------- | ---------- | ------ | -------- | ----- | ------ | -------- | ------------------- | ----------- | ------------ | ---------- | ------------------ |
| NCCL    | 2.7.8   | \-            | 8          | 1      | RTX3090  | 32    | 200    | 7.026861 | 1126.93             | 64          | \-           | scratch    | \-                 |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 8     | 200    | 7.026847 | 1612.23             | 64          | 1024         | scratch    | 1.43063899266148 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 6     | 200    | 7.026935 | 1532.99             | 64          | 1024         | scratch    | 1.36032406626853 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 5     | 200    | 7.027552 | 1560.79             | 64          | 1024         | scratch    | 1.38499285669917 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 4     | 200    | 7.029195 | 2112.71             | 64          | 1024         | scratch    | 1.87474820973796 x |



### dm_nfnet_f1 on ImageNet

| backend | version | quant mode    | #GPUs/node | #nodes | GPU arch | #BITS | #Steps | loss     | throughput(img/sec) | batch\_size | bucket\_size | train type | speedup            |
| ------- | ------- | ------------- | ---------- | ------ | -------- | ----- | ------ | -------- | ------------------- | ----------- | ------------ | ---------- | ------------------ |
| NCCL    | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 32    | 200    | 6.90869  | 364.78              | 64          | \-           | scratch    | \-                 |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 8     | 200    | 6.908694 | 485.37              | 64          | 1024         | scratch    | 1.33058281704041 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 6     | 200    | 6.90871  | 479.91              | 64          | 1024         | scratch    | 1.31561489116728 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 5     | 200    | 6.908757 | 486.93              | 64          | 1024         | scratch    | 1.33485936728987 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 4     | 200    | 6.908916 | 616.12              | 64          | 1024         | scratch    | 1.68901803826964 x |


### dm_nfnet_f3 on ImageNet

| backend | version | quant mode    | #GPUs/node | #nodes | GPU arch | #BITS | #Steps | loss     | throughput(img/sec) | batch\_size | bucket\_size | train type | speedup            |
| ------- | ------- | ------------- | ---------- | ------ | -------- | ----- | ------ | -------- | ------------------- | ----------- | ------------ | ---------- | ------------------ |
| NCCL    | 2.7.8   | \-            | 8          | 1      | RTX3090  | 32    | 200    | 6.906141 | 53.97               | 16          | \-           | scratch    | \-                 |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 8     | 200    | 6.90614  | 75.22               | 16          | 1024         | scratch    | 1.39373726144154 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 6     | 200    | 6.906144 | 78.6                | 16          | 1024         | scratch    | 1.45636464702613 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 5     | 200    | 6.906156 | 76.29               | 16          | 1024         | scratch    | 1.41356309060589 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 4     | 200    | 6.906216 | 105.09              | 16          | 1024         | scratch    | 1.94719288493608 x |


### Resnext 50_32*4d on ImageNet

| backend | version | quant mode    | #GPUs/node | #nodes | GPU arch | #BITS | #Steps | loss   | throughput(img/sec) | batch\_size | bucket\_size | train type | speedup            |
| ------- | ------- | ------------- | ---------- | ------ | -------- | ----- | ------ | ------ | ------------------- | ----------- | ------------ | ---------- | ------------------ |
| NCCL    | 2.7.8   | \-            | 8          | 1      | RTX3090  | 32    | 200    | 6.9001 | 152.050689          | 32          | \-           | scratch    | \-                 |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 8     | 200    | 6.8991 | 230.6697441         | 32          | 1024         | scratch    | 1.51705819774827 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 6     | 200    | 6.8976 | 250.5581332         | 32          | 1024         | scratch    | 1.64785924301116 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 5     | 200    | 6.8991 | 243.5203971         | 32          | 1024         | scratch    | 1.60157378310316 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 4     | 200    | 6.904  | 300.1660043         | 32          | 1024         | scratch    | 1.97411801540888 x |


### VGG16 on ImageNet

| backend | version | quant mode    | #GPUs/node | #nodes | GPU arch | #BITS | #Steps | loss   | throughput(img/sec) | batch\_size | bucket\_size | train type | speedup            |
| ------- | ------- | ------------- | ---------- | ------ | -------- | ----- | ------ | ------ | ------------------- | ----------- | ------------ | ---------- | ------------------ |
| NCCL    | 2.7.8   | \-            | 8          | 1      | RTX3090  | 32    | 200    | 6.9054 | 26.7526028          | 32          | \-           | scratch    | \-                 |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 8     | 200    | 6.906  | 63.41101874         | 32          | 1024         | scratch    | 2.37027474321221 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 6     | 200    | 6.9065 | 64.82861472         | 32          | 1024         | scratch    | 2.42326382968653 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 5     | 200    | 6.9067 | 64.30133894         | 32          | 1024         | scratch    | 2.40355450327789 x |
| QNCCL   | 2.7.8   | deterministic | 8          | 1      | RTX3090  | 4     | 200    | 6.9058 | 105.5437996         | 32          | 1024         | scratch    | 3.94517873235204 x |





