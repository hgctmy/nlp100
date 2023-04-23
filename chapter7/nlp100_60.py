from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)  # モデルを読み込む
print(model['United_States'])  # ベクトル表現を表示

'''
% python nlp100_60.py
[-3.61328125e-02 -4.83398438e-02  2.35351562e-01  1.74804688e-01
 -1.46484375e-01 -7.42187500e-02 -1.01562500e-01 -7.71484375e-02
  1.09375000e-01 -5.71289062e-02 -1.48437500e-01 -6.00585938e-02
  1.74804688e-01 -7.71484375e-02  2.58789062e-02 -7.66601562e-02
 -3.80859375e-02  1.35742188e-01  3.75976562e-02 -4.19921875e-02
 -3.56445312e-02  5.34667969e-02  3.68118286e-04 -1.66992188e-01
 -1.17187500e-01  1.41601562e-01 -1.69921875e-01 -6.49414062e-02
 -1.66992188e-01  1.00585938e-01  1.15722656e-01 -2.18750000e-01
 -9.86328125e-02 -2.56347656e-02  1.23046875e-01 -3.54003906e-02
 -1.58203125e-01 -1.60156250e-01  2.94189453e-02  8.15429688e-02
  6.88476562e-02  1.87500000e-01  6.49414062e-02  1.15234375e-01
 -2.27050781e-02  3.32031250e-01 -3.27148438e-02  1.77734375e-01
 -2.08007812e-01  4.54101562e-02 -1.23901367e-02  1.19628906e-01
  7.44628906e-03 -9.03320312e-03  1.14257812e-01  1.69921875e-01
 -2.38281250e-01 -2.79541016e-02 -1.21093750e-01  2.47802734e-02
  7.71484375e-02 -2.81982422e-02 -4.71191406e-02  1.78222656e-02
 -1.23046875e-01 -5.32226562e-02  2.68554688e-02 -3.11279297e-02
 -5.59082031e-02 -5.00488281e-02 -3.73535156e-02  1.25976562e-01
  5.61523438e-02  1.51367188e-01  4.29687500e-02 -2.08007812e-01
 -4.78515625e-02  2.78320312e-02  1.81640625e-01  2.20703125e-01
 -3.61328125e-02 -8.39843750e-02 -3.69548798e-05 -9.52148438e-02
 -1.25000000e-01 -1.95312500e-01 -1.50390625e-01 -4.15039062e-02
  1.31835938e-01  1.17675781e-01  1.91650391e-02  5.51757812e-02
 -9.42382812e-02 -1.08886719e-01  7.32421875e-02 -1.15234375e-01
  8.93554688e-02 -1.40625000e-01  1.45507812e-01  4.49218750e-02
 -1.10473633e-02 -1.62353516e-02  4.05883789e-03  3.75976562e-02
 -6.98242188e-02 -5.46875000e-02  2.17285156e-02 -9.47265625e-02
  4.24804688e-02  1.81884766e-02 -1.73339844e-02  4.63867188e-02
 -1.42578125e-01  1.99218750e-01  1.10839844e-01  2.58789062e-02
 -7.08007812e-02 -5.54199219e-02  3.45703125e-01  1.61132812e-01
 -2.44140625e-01 -2.59765625e-01 -9.71679688e-02  8.00781250e-02
 -8.78906250e-02 -7.22656250e-02  1.42578125e-01 -8.54492188e-02
 -3.18359375e-01  8.30078125e-02  6.34765625e-02  1.64062500e-01
 -1.92382812e-01 -1.17675781e-01 -5.41992188e-02 -1.56250000e-01
 -1.21582031e-01 -4.95605469e-02  1.20117188e-01 -3.83300781e-02
  5.51757812e-02 -8.97216797e-03  4.32128906e-02  6.93359375e-02
  8.93554688e-02  2.53906250e-01  1.65039062e-01  1.64062500e-01
 -1.41601562e-01  4.58984375e-02  1.97265625e-01 -8.98437500e-02
  3.90625000e-02 -1.51367188e-01 -8.60595703e-03 -1.17675781e-01
 -1.97265625e-01 -1.12792969e-01  1.29882812e-01  1.96289062e-01
  1.56402588e-03  3.93066406e-02  2.17773438e-01 -1.43554688e-01
  6.03027344e-02 -1.35742188e-01  1.16210938e-01 -1.59912109e-02
  2.79296875e-01  1.46484375e-01 -1.19628906e-01  1.76757812e-01
  1.28906250e-01 -1.49414062e-01  6.93359375e-02 -1.72851562e-01
  9.22851562e-02  1.33056641e-02 -2.00195312e-01 -9.76562500e-02
 -1.65039062e-01 -2.46093750e-01 -2.35595703e-02 -2.11914062e-01
  1.84570312e-01 -1.85546875e-02  2.16796875e-01  5.05371094e-02
  2.02636719e-02  4.25781250e-01  1.28906250e-01 -2.77099609e-02
  1.29882812e-01 -1.15722656e-01 -2.05078125e-02  1.49414062e-01
  7.81250000e-03 -2.05078125e-01 -8.05664062e-02 -2.67578125e-01
 -2.29492188e-02 -8.20312500e-02  8.64257812e-02  7.61718750e-02
 -3.66210938e-02  5.22460938e-02 -1.22070312e-01 -1.44042969e-02
 -2.69531250e-01  8.44726562e-02 -2.52685547e-02 -2.96630859e-02
 -1.68945312e-01  1.93359375e-01 -1.08398438e-01  1.94091797e-02
 -1.80664062e-01  1.93359375e-01 -7.08007812e-02  5.85937500e-02
 -1.01562500e-01 -1.31835938e-01  7.51953125e-02 -7.66601562e-02
  3.37219238e-03 -8.59375000e-02  1.25000000e-01  2.92968750e-02
  1.70898438e-01 -9.37500000e-02 -1.09375000e-01 -2.50244141e-02
  2.11914062e-01 -4.44335938e-02  6.12792969e-02  2.62451172e-02
 -1.77734375e-01  1.23046875e-01 -7.42187500e-02 -1.67968750e-01
 -1.08886719e-01 -9.04083252e-04 -7.37304688e-02  5.49316406e-02
  6.03027344e-02  8.39843750e-02  9.17968750e-02 -1.32812500e-01
  1.22070312e-01 -8.78906250e-03  1.19140625e-01 -1.94335938e-01
 -6.64062500e-02 -2.07031250e-01  7.37304688e-02  8.93554688e-02
  1.81884766e-02 -1.20605469e-01 -2.61230469e-02  2.67333984e-02
  7.76367188e-02 -8.30078125e-02  6.78710938e-02 -3.54003906e-02
  3.10546875e-01 -2.42919922e-02 -1.41601562e-01 -2.08007812e-01
 -4.57763672e-03 -6.54296875e-02 -4.95605469e-02  2.22656250e-01
  1.53320312e-01 -1.38671875e-01 -5.24902344e-02  4.24804688e-02
 -2.38281250e-01  1.56250000e-01  5.83648682e-04 -1.20605469e-01
 -9.22851562e-02 -4.44335938e-02  3.61328125e-02 -1.86767578e-02
 -8.25195312e-02 -8.25195312e-02 -4.05273438e-02  1.19018555e-02
  1.69921875e-01 -2.80761719e-02  3.03649902e-03  9.32617188e-02
 -8.49609375e-02  1.57470703e-02  7.03125000e-02  1.62353516e-02
 -2.27050781e-02  3.51562500e-02  2.47070312e-01 -2.67333984e-02]
higuchitomoya@MacBook-Air chapter7 % 
'''
