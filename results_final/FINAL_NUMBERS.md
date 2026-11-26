# Final system: out-of-fold numbers

998 clips (500 real, 498 fake), 500 speakers, 5-fold speaker-disjoint CV.

AUC voice 0.817, face 0.746, fused 0.9. Coverage 83.8%; accuracy on answered clips 86.6%; 162 clips UNCERTAIN.

## 3-way confusion

```
verdict      REAL  FAKE  UNCERTAIN
truth                             
REAL (true)   366    57         77
FAKE (true)    55   358         85
```

## Class-wise

```
class  support  predicted  true_positives  precision  recall_on_answered  recall_incl_uncertain  f1_on_answered  uncertain
 REAL      500        421             366      0.869               0.865                  0.732           0.867         77
 FAKE      498        415             358      0.863               0.867                  0.719           0.865         85
```

## Per category

```
category                  meaning  clips  REAL  FAKE  UNCERTAIN  correct_of_answered  correct_of_all
    RVRA  real video + real audio    500   366    57         77                0.865           0.732
    RVFA             cloned voice    166     1   128         37                0.992           0.771
    FVRA                face swap    166    54    69         43                0.561           0.416
    FVFA face swap + cloned voice    166     0   161          5                1.000           0.970
```

## Sanity checks

```
                                                          test  pairs  REAL  FAKE  UNCERTAIN expected
                           genuine clip (own voice + own face)    500   366    57         77     REAL
real voice + real face of ANOTHER person (mismatched identity)    500   353    67         80     REAL
                                      cloned voice + real face    166     1   126         39     FAKE
                                     real voice + swapped face    500   149   239        112     FAKE
                                   cloned voice + swapped face    166     0   159          7     FAKE
```

## Bias: ethnicity

```
               group  clips  speakers  real  fake  accuracy_answered  false_alarm_rate  miss_rate  uncertain_rate
             African    198       100   100    98              0.882             0.088      0.151           0.227
        Asian (East)    184       100   100    84              0.816             0.165      0.209           0.174
       Asian (South)    203       100   100   103              0.861             0.101      0.176           0.113
Caucasian (American)    202       100   100   102              0.874             0.212      0.046           0.173
Caucasian (European)    211       100   100   111              0.891             0.112      0.105           0.128
```

## Bias: sex

```
group  clips  speakers  real  fake  accuracy_answered  false_alarm_rate  miss_rate  uncertain_rate
  men    521       250   250   271              0.839             0.177      0.146           0.154
women    477       250   250   227              0.896             0.091      0.118           0.172
```