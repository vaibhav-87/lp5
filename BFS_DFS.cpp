#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
using namespace std;
using namespace chrono;

int visited[1000], visit[1000];
int que[1000], front = 0, rear = 0;
int stk[1000], top = -1;

void bfs_seq(int cost[1000][1000], int n, int start) {
    fill(visited, visited + n, 0);
    fill(visit, visit + n, 0);
    front = rear = 0;

    cout << "BFS Sequential: ";
    visited[start] = 1;
    cout << start << " ";
    que[rear++] = start;

    while (rear > front) {
        int v = que[front++];
        for (int j = 0; j < n; j++) {
            if (cost[v][j] && !visited[j] && !visit[j]) {
                visit[j] = visited[j] = 1;
                que[rear++] = j;
                cout << j << " ";
            }
        }
    }
    cout << endl;
}

void bfs_par(int cost[1000][1000], int n, int start) {
    fill(visited, visited + n, 0);
    fill(visit, visit + n, 0);
    front = rear = 0;

    cout << "BFS Parallel: ";
    visited[start] = 1;
    cout << start << " ";
    que[rear++] = start;

    while (rear > front) {
        int v = que[front++];
        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            if (cost[v][j] && !visited[j] && !visit[j]) {
                #pragma omp critical
                {
                    visit[j] = visited[j] = 1;
                    que[rear++] = j;
                    cout << j << " ";
                }
            }
        }
    }
    cout << endl;
}

void dfs_seq(int cost[1000][1000], int n, int start) {
    fill(visited, visited + n, 0);
    fill(visit, visit + n, 0);
    top = -1;

    cout << "DFS Sequential: ";
    visited[start] = 1;
    cout << start << " ";
    stk[++top] = start;

    while (top >= 0) {
        int v = stk[top--];
        for (int j = n - 1; j >= 0; j--) {
            if (cost[v][j] && !visited[j] && !visit[j]) {
                visit[j] = visited[j] = 1;
                stk[++top] = j;
                cout << j << " ";
            }
        }
    }
    cout << endl;
}

void dfs_par(int cost[1000][1000], int n, int start) {
    fill(visited, visited + n, 0);
    fill(visit, visit + n, 0);
    top = -1;

    cout << "DFS Parallel: ";
    visited[start] = 1;
    cout << start << " ";
    stk[++top] = start;

    while (top >= 0) {
        int v = stk[top--];
        #pragma omp parallel for
        for (int j = n - 1; j >= 0; j--) {
            if (cost[v][j] && !visited[j] && !visit[j]) {
                #pragma omp critical
                {
                    visit[j] = visited[j] = 1;
                    stk[++top] = j;
                    cout << j << " ";
                }
            }
        }
    }
    cout << endl;
}

int main() {
    int n = 200;
    int cost[1000][1000];

    // Create symmetric random graph
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            cost[i][j] = cost[j][i] = (i != j) ? rand() % 2 : 0;
        }
    }

    int start;
    cout << "Enter starting vertex for BFS: ";
    cin >> start;

    auto t1 = high_resolution_clock::now();
    bfs_seq(cost, n, start);
    auto t2 = high_resolution_clock::now();
    bfs_par(cost, n, start);
    auto t3 = high_resolution_clock::now();

    cout << "\nEnter starting vertex for DFS: ";
    cin >> start;
    auto t4 = high_resolution_clock::now();
    dfs_seq(cost, n, start);
    auto t5 = high_resolution_clock::now();
    dfs_par(cost, n, start);
    auto t6 = high_resolution_clock::now();

    auto bfsSeqTime = duration_cast<microseconds>(t2 - t1);
    auto bfsParTime = duration_cast<microseconds>(t3 - t2);
    auto dfsSeqTime = duration_cast<microseconds>(t5 - t4);
    auto dfsParTime = duration_cast<microseconds>(t6 - t5);

    cout << "\nTime Taken (BFS Sequential): " << bfsSeqTime.count() << " microseconds\n";
    cout << "Time Taken (BFS Parallel): "   << bfsParTime.count() << " microseconds\n";
    cout << "Time Taken (DFS Sequential): " << dfsSeqTime.count() << " microseconds\n";
    cout << "Time Taken (DFS Parallel): "   << dfsParTime.count() << " microseconds\n";

    return 0;
}
