import sys
import numpy as np
import json
import pandas as pd
import collections

if __name__ == '__main__':

    for epoch in [20, 100]:
        print(epoch)
        real_res = {'logreg':[-1], 'svc':[-1], 'linearsvc':[-1], 'randomforest':[-1]}
        for gc in [3, 5, 8, 16]:
            for lr in [0.01, 0.1, 0.001]:
                for tpe in ['local', 'localprior']:
                    res = collections.defaultdict(lambda :collections.defaultdict(list))
                    with open(sys.argv[1], 'r') as f:
                        for line in f:
                            x = line.strip().split(',', 6)
                            if x[1] != tpe:
                                continue
                            if x[2] != str(gc):
                                continue
                            if x[3] != str(epoch):
                                continue
                            if x[5] != str(lr):
                                continue
                            tmp = json.loads(x[-1])

                            DS = x[0]
                            res[DS]['logreg'].append(tmp['logreg'])
                            res[DS]['svc'].append(tmp['svc'])
                            res[DS]['linearsvc'].append(tmp['linearsvc'])
                            res[DS]['randomforest'].append(tmp['randomforest'])

                    for DS, lst in res.items():
                        if DS != sys.argv[2]:
                            continue
                        # print('====================')
                        # print(DS)
                        for clf, v in lst.items():
                            mn = np.mean(np.array(v[:5]), axis=0)
                            std = np.std(np.array(v[:5]), axis=0)

                            idx = np.argmax(mn)
                            if mn[idx] > real_res[clf][0] and len(v) > 1:
                                real_res[clf] = [mn[idx], std[idx], epoch, lr, gc, idx, len(v)]
                                # print(epoch, lr, gc, clf, idx, mn[idx], std[idx], len(v))
        print(real_res)

