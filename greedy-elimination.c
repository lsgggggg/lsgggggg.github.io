#include <stdio.h>

#define MAXN 105

int n;
int g[MAXN][MAXN];               /* g[行][列]，1 起始 */
long sumP[MAXN][MAXN];           /* 分数前缀和 */
int zeroP[MAXN][MAXN];           /* 黑洞个数前缀和 */

int bx1, by1, bx2, by2;          /* x 为列、y 为行 */
long best;

void build_prefix(void) {
    for (int r = 0; r <= n; r++)
        for (int c = 0; c <= n; c++) { sumP[r][c] = 0; zeroP[r][c] = 0; }
    for (int r = 1; r <= n; r++)
        for (int c = 1; c <= n; c++) {
            sumP[r][c] = g[r][c] + sumP[r-1][c] + sumP[r][c-1] - sumP[r-1][c-1];
            zeroP[r][c] = (g[r][c] == 0) + zeroP[r-1][c] + zeroP[r][c-1] - zeroP[r-1][c-1];
        }
}

long rect_sum(int y1, int x1, int y2, int x2) {
    return sumP[y2][x2] - sumP[y1-1][x2] - sumP[y2][x1-1] + sumP[y1-1][x1-1];
}
int rect_zero(int y1, int x1, int y2, int x2) {
    return zeroP[y2][x2] - zeroP[y1-1][x2] - zeroP[y2][x1-1] + zeroP[y1-1][x1-1];
}

void find_best(void) {
    best = -1000000000L;
    bx1 = -1;
    for (int x1 = 1; x1 <= n; x1++)
        for (int y1 = 1; y1 <= n; y1++)
            for (int x2 = x1; x2 <= n; x2++)
                for (int y2 = y1; y2 <= n; y2++) {
                    if (rect_zero(y1, x1, y2, x2)) break;   /* 含黑洞，再扩也含，停 */
                    long s = rect_sum(y1, x1, y2, x2);
                    if (s > best) { best = s; bx1 = x1; by1 = y1; bx2 = x2; by2 = y2; }
                }
}

void apply_move(void) {
    int h = by2 - by1 + 1;
    for (int c = bx1; c <= bx2; c++)
        for (int r = by2; r >= 1; r--)
            g[r][c] = (r - h >= 1) ? g[r-h][c] : 0;   /* 上方下落，顶部补黑洞 */
}

int main(void) {
    scanf("%d", &n);
    for (int r = 1; r <= n; r++)
        for (int c = 1; c <= n; c++)
            scanf("%d", &g[r][c]);

    long total = 0;
    while (1) {
        build_prefix();
        find_best();
        if (best <= 0) break;
        printf("(%d, %d) (%d, %d) %ld\n", bx1, by1, bx2, by2, best);
        total += best;
        apply_move();
    }
    printf("%ld\n", total);
    return 0;
}
