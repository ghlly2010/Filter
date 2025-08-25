# Filter

## 1. 필터란란
- 신호에서 생기는 **노이즈를 제거**하는 역할을 한다.
- 대표적 예시 : MAF IIR 필터 FIR 필터
## 2. 필터의 종류

### 1. Low pass Filter
<br>
<div align="center">
<img width="400"  src="https://github.com/user-attachments/assets/cf91f482-a011-41c9-bb04-fac85d2e586b" />
</div>
<br>
Low pass Filter는 낮은 주파수 영역의 신호만 통과시키는 필터이다.

### 2. High pass Filter

<br>
<div align="center">
<img width="400"  src="https://github.com/user-attachments/assets/7ee29e80-d938-4653-92dc-c3fe0356b93c" />
</div>
<br>

High pass Filter는 높은 주파수 영역의 신호만 통과시키는 필터이다.

### 3. band pass Filter
<br>
<div align="center">
<img width="400"  src="https://github.com/user-attachments/assets/d2606146-4df8-42d8-b22d-39d523dd2aa3" />
</div>
<br>
band pass Filter는 일정한 구간 내의 신호를 통과시키는 필터이다.

### 4. band stop Filter
<br>
<div align="center">
<img width="400" src="https://github.com/user-attachments/assets/09e8e7c7-d583-44f1-9fd0-8da826140e2d" />
</div>
<br>
band stop Filter는 일정한 구간 이외의 신호를 통과시키는 필터이다.
<br>


## 3. Digital Filter
- 디지털 필터는 아날로그 필터의 성능을 높이기 위해 고안된 필터이다.
### 1. MAF(Moving Average Filter)
- **Moving Average 필터** (이동 평균 필터)는 가장 간단하면서도 널리 사용되는 디지털 필터 중 하나이다. 
  <br>
- 일정 개수의 최근 입력 신호 값을 평균 내어 출력으로 사용하는 방식으로, **노이즈 제거**와 **신호 평활화(Smoothing)** 에 효과적이다.

#### 1. 수식표현
윈도우 크기가 `N`일 때, 시점 `n`에서의 출력 `y[n]`은:

$$
y[n] = \frac{1}{N} \sum_{k=0}^{N-1} x[n-k]
$$

여기서,
- \( x[n] \) : 입력 신호
- \( y[n] \) : 출력 신호
- \( N \) : 윈도우 크기


#### 2. 장점
- 구현이 매우 간단하다.
- 고주파 잡음을 효과적으로 제거할 수 있다.
- 선형 위상 특성을 가진다 (위상 왜곡이 없음).

####  3. 단점
- 신호의 급격한 변화(에지, 피크)를 희석시켜 손실을 유발할 수 있다.
- 모든 주파수 성분을 동일하게 평활화하여, 원하는 대역만 선택적으로 필터링하기 어렵다.
- 큰 윈도우 크기를 사용할수록 반응 속도가 느려진다.

#### 4. 코드 예시

##### 파이썬 코드
```python
import numpy as np

def moving_average(x, N):
    return np.convolve(x, np.ones(N)/N, mode='valid')

# 예제 데이터
data = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 윈도우 크기 3으로 이동 평균 적용
filtered = moving_average(data, 3)
print(filtered)  # [2. 3. 4. 5. 6. 7. 8.]

```
##### C언어 코드
```c
#include<stdio.h>
int main()
{
    int arr[10]={0};        //데이터를 저장할 배열
    for(int i=0;i<10;i++)
    {
        int sum=0;         // 합을 저장할 변수
        arr[i]=input_data; //입력할 데이터를 배열에 저장
        for(int j=0;j<10;j++)
        {
            sum+=arr[j];   // 배열의 요소를 하나씩 더함
        } 
        int Filter=sum/10; // 필터링 결과
    }
    return 0;
}
```

### 2. IIR Filter(Infinite Impulse Response Filter)
- **IIR 필터**는 "무한 임펄스 응답 필터"라는 뜻으로, 출력이 과거의 입력뿐만 아니라 **과거의 출력 값**에도 의존하는 디지털 필터이다. 이로 인해 필터의 임펄스 응답이 이론적으로 **무한히 지속**되는 특징을 가진다. 
- 출력이 과거의 입력과 출력 모두에 의존하기에 임펄스 응답이 무한대이다.

#### 1. 수식표현
일반적인 IIR 필터는 다음과 같은 **차분방정식(difference equation)**으로 표현된다.

$$
y[n] = \sum_{k=0}^{M} b_k \cdot x[n-k] - \sum_{k=1}^{N} a_k \cdot y[n-k]
$$

여기서,  
- \( x[n] \): 입력 신호  
- \( y[n] \): 출력 신호  
- \( b_k \): 분자 계수 (Feedforward 계수)  
- \( a_k \): 분모 계수 (Feedback 계수)  
- \( M \): FIR 부분의 차수  
- \( N \): IIR 부분의 차수  

#### 2. 장점
- **효율적**: 같은 주파수 응답을 얻을 때 FIR보다 차수가 낮아 연산량이 적다.
- **메모리 절약**: 적은 계수로 원하는 필터 특성을 구현 가능.
- **아날로그 필터와 유사**: Butterworth, Chebyshev, Elliptic 등의 아날로그 필터를 디지털로 구현 가능.

#### 3. 단점
- **위상 왜곡** 발생 가능 (선형 위상이 아님).
- 시스템이 불안정해질 수 있음 (폴이 단위원(Unit circle) 밖에 존재하면 불안정).
- 구현 및 설계가 FIR보다 복잡.

#### 4. 주파수 응답
IIR 필터는 전달 함수(Transfer Function)로 표현할 수 있다.

\[
H(z) = \frac{\sum_{k=0}^{M} b_k z^{-k}}{1 + \sum_{k=1}^{N} a_k z^{-k}}
\]

- 분모(denominator)가 존재하기 때문에 **무한 응답**이 발생.
- 폴(pole)과 제로(zero)의 위치에 따라 주파수 특성이 결정됨.


#### 5. 코드 예시
##### 파이썬 코드
```python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Butterworth 저역통과 IIR 필터 설계
b, a = signal.butter(N=4, Wn=0.2)  # 차수 4, 차단 주파수 0.2 (정규화)

# 주파수 응답
w, h = signal.freqz(b, a)

plt.plot(w/np.pi, 20*np.log10(abs(h)))
plt.title("IIR Filter (Butterworth) Frequency Response")
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude (dB)")
plt.grid()
plt.show()
```
##### C언어 코드
``` c
#include<stdio.h>
int main()
{
    double a=0.2; // 필터계수
    double x[10]; //입력신호
    double y[10]; //출력신호
    y[0]=a*x[0];

    for (int n = 1; n < 10; n++) {
        y[n] = a * x[n] + (1.0 - a) * y[n - 1];
    }
    return 0;
}
```
### 3. FIR Filter(Finite Impulse Response Filter)
- **FIR 필터**는 "유한 임펄스 응답 필터"라는 뜻으로, 출력이 **과거의 입력 값들에만 의존**하는 디지털 필터이다.  
임펄스 응답이 일정 길이 이후에 0이 되기 때문에 "유한"이라고 부른다.  
- 출력이 과거의 입력 값에만 의존하기에 임펄스 응답이 유한하다.

#### 1. 수식표현현
FIR 필터의 출력은 다음과 같은 **차분방정식**으로 정의된다:

$$
y[n] = \sum_{k=0}^{M} b_k \cdot x[n-k]
$$

여기서,  
- \( x[n] \): 입력 신호  
- \( y[n] \): 출력 신호  
- \( b_k \): 필터 계수 (임펄스 응답)  
- \( M \): 필터 차수 (계수 개수 - 1)  

즉, 출력은 입력 샘플들의 **가중합(Weighted Sum)** 으로 표현된다.  

#### 2. 장점
- 항상 **안정적**이다 (피드백이 없기 때문).  
- **선형 위상(Linear Phase)** 설계 가능 → 신호의 파형 왜곡이 적다.  
- 구현이 단순하고 직관적이다.  

#### 3. 단점
- 동일한 성능을 얻기 위해서는 **IIR보다 높은 차수**가 필요 → 연산량 증가.  
- 실시간 시스템에서는 계산량이 부담될 수 있다.  

---

#### 4. 주파수 응답
FIR 필터의 전달 함수는 다음과 같이 표현된다:

\[
H(z) = \sum_{k=0}^{M} b_k z^{-k}
\]

즉, 분모(denominator)가 없으므로 **유한 임펄스 응답**이 된다.  




#### 5. 코드 예시
##### 파이썬 코드
```python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# FIR 저역통과 필터 설계 (윈도우법, Hamming)
numtaps = 21   # 필터 계수 개수
cutoff = 0.3   # 정규화 차단 주파수
b = signal.firwin(numtaps, cutoff, window="hamming")

# 주파수 응답
w, h = signal.freqz(b)

plt.plot(w/np.pi, 20*np.log10(abs(h)))
plt.title("FIR Filter (Low-pass, Hamming Window)")
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude (dB)")
plt.grid()
plt.show()
```
##### C언어 코드
```c
#include<stdio.h>
int main()
{
    double b[5]={0.3,0.3,0.3,0.3,0.3}; // 필터계수 5
    double x[10]; //입력신호
    double y[10]; //출력신호
    y[0]=a*x[0];

    for (int n = 0; n < 10; n++) {
        for (int k = 0; k < 5; k++) {
            if (n - k >= 0) {
                y[n] += b[k] * x[n - k];
            }
        }
    }
    return 0;
}
```

  
