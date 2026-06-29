#include <stdio.h>
#include <string.h>

#define MAXN 10005

int arrive[MAXN];        /* 下标为预约号 */
char pid[MAXN][6];       /* 5 位 ID 字符串，保留前导零 */
int age[MAXN];
int done[MAXN];

int main(void) {
    int n;
    scanf("%d", &n);
    for (int i = 0; i < n; i++) {
        int a, p, ag;
        char s[6];
        scanf("%d %d %s %d", &a, &p, s, &ag);
        arrive[p] = a;
        strcpy(pid[p], s);
        age[p] = ag;
    }

    int treated = 0, t = 1;
    while (treated < n) {
        int pick = -1;
        if (t <= n && !done[t] && arrive[t] <= t) {   /* 当前号槽的患者正在等待 */
            pick = t;
        } else {
            for (int b = 1; b <= n; b++)              /* 老人优先，取预约号最小 */
                if (!done[b] && arrive[b] <= t && age[b] >= 80) { pick = b; break; }
            if (pick == -1)
                for (int b = 1; b <= n; b++)          /* 否则取等待者中预约号最小 */
                    if (!done[b] && arrive[b] <= t) { pick = b; break; }
        }
        if (pick != -1) {
            printf("%d %s\n", t, pid[pick]);
            done[pick] = 1;
            treated++;
        }
        t++;
    }
    return 0;
}
