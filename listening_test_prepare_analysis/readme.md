ここでは，聴取実験の評価データの作成，評価結果の分析を行います．

1. ノイズ信号の推定
- train.py, train.sh, utils.sh
- 発話データごとに，電子透かしと類似した特性を持つノイズ信号を勾配降下により推定します
- 必ずしも目標とする客観評価値に達成するという保証がないため，パラメータなどを適宜変更する必要があります

2. 後処理
- amp_normalize.ipynb
- 聴取実験用の評価データに対して振幅正規化を行います
- また，発話ごとのディレクトリを，ノイズのクラスごとに再構成します
- さらに，それらの情報を安田先生のオンライン聴取実験システム上で利用できるよう，csvファイルを作成します
- ここまでで作られたデータは nas02/internal/d_yoon/listening_test_prepare_analysis の下をご参照ください

3. 実験データの視覚化
- optimized_noise.ipynb
- 作られた評価データに対して，SNR・GDをグラフで表示します（修論の図3.1）

4. 聴取実験実施
- 安田先生のシステムのマニュアルに従って主観評価（DMOS，動的比較評価）を行います
- 使用したコードは nas02/internal/d_yoon/nii-listening-test-scripts の下をご参照ください
- 実験結果のjsonファイル（test_score-*.json）を分析に使用します

5. DMOS評価分析
- results_dmos/score.ipynb (, results_dmos/dmos-pesq.ipynb)
- 結果から不正な被験者のデータを除きます
- ノイズクラスごとのDMOS評点を視覚化します（修論の図3.2）
- 線型回帰を用いて，DMOSの予測式を推定します
- PESQスコアやLAQスコア（動的比較評価の絶対値スコア）とも比較できるようにしていますが，修論には載っていないです

6. 動的比較評価分析
- results_compare/score.ipynb (, results_compare/results_to_sample.ipynb)
- 動的比較評価をヒートマップの形式に視覚化します（修論の図3.3）
- LAQスコアの計算および視覚化もできるようにしていますが，修論には載っていないです

7. ランキング相関計算
- ranking_corr.ipynb
- DMOS・動的比較評価結果のランキング相関を計算し，視覚化します（修論の図3.4）