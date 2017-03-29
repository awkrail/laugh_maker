# laugh_maker

---

laugh_makerは研究用のリポジトリです.<br>
ニューラルネットワークを使って分類をすることを考えています.<br>

```
chainer 1.18.0
python 3.5.1
anaconda3
```
をインストールしてください.<br>

## predict

学習したデータをpredictする際は,predict.pyを起動してください.<br>
現在,どういう風に本番データを渡すようにするのかを考え中です.<br>

## results

現行の結果です.<br>
(03/28),(3/29)<br>
L,H,Rawデータの三種類を考察<br>
10996の笑っているときの1秒あたりのデータを0.1秒刻みとして10次元のベクトルを以下のように分類する<br>
<br>
(1)3層のニューラルネットワーク<br>
(2)LSTM層を含む,3層のニューラルネットワーク<br>
(3)6層のニューラルネットワーク<br>
(4)LSTM層を含む,LデータとHデータを合わせて(2*10)の行列にしたものを入れるニューラルネットワーク<br>
<br>
L,Hのデータに関しては,(1)~(4)すべてについて70-75%の結果となった.<br>
Rawデータの場合,(1)~(3)のネットワークで検証を行ったところ,全て79-80%という結果となった.<br>
<br>
以上より,Rawデータで今後実験を重ねていこうと考えている.<br>
(1),(3)の結果がほぼ同じであったことから,ニューラルネットの表現力は十分であるものと考えられ,<br>
データの説明変数の少なさ(もう少し入力の次元を大きくする)か単純にデータの量を増やすことが精度を上げる手段であると考えられる
