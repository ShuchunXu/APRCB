```
python ./main.py
```
natural training+balance
```
python ./main.py --imbalance=1.0 --train_type=natural --num_epochs=num_epochs
```
adversarial training+balance
```
python ./main.py --imbalance=1.0 --train_type=adversarial --num_epochs=num_epochs
```
natural training+long-tail
```
python ./main.py --imbalance=0.1 --train_type=natural --num_epochs=num_epochs
```
adversarial training+long-tail
```
python ./main.py --imbalance=0.1 --train_type=adversarial --num_epochs=num_epochs
```
adversarial training+long-tail+Our's method
```
python ./main.py --imbalance=0.1 --train_type=adversarial_ours --num_epochs=num_epochs
```
