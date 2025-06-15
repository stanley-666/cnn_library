# build

```bash
git clone git@github.com:stanley-666/cnn_library.git
chmod +x build.sh
./build.sh
````

# pytools

## col.py
把weight.h的array定義複製進去成string，然後輸出會是python的array定義，再把輸出複製到reshape2dto4d.py
## reshape2dto4d.py
這樣可以把2d權重轉換成4d [OutC][InC][kernel][kernel]的格式，然後順便輸出成C的格式到converted_weight.h裡面，給main.c做使用。
