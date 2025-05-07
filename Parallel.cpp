#include <iostream>
#include <vector>
#include <climits>
#include <cstdlib>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

void findMinimum(const vector<int>& array) {
    int min_val = INT_MAX;
    auto start = high_resolution_clock::now();
    for (int i = 0; i < array.size(); ++i) {
        if (array[i] < min_val) {
            min_val = array[i];
        }
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "Minimum Element: " << min_val << endl;
    cout << "Time Taken: " << duration.count() << " microseconds\n";

    int min_parallel = INT_MAX;
    start = high_resolution_clock::now();
    #pragma omp parallel for reduction(min:min_parallel)
    for (int i = 0; i < array.size(); ++i) {
        if (array[i] < min_parallel) {
            min_parallel = array[i];
        }
    }
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);
    cout << "Minimum Element (Parallel): " << min_parallel << endl;
    cout << "Time Taken: " << duration.count() << " microseconds\n\n";
}

void findMaximum(const vector<int>& array) {
    int max_val = INT_MIN;
    auto start = high_resolution_clock::now();
    for (int i = 0; i < array.size(); ++i) {
        if (array[i] > max_val) {
            max_val = array[i];
        }
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "Maximum Element: " << max_val << endl;
    cout << "Time Taken: " << duration.count() << " microseconds\n";

    int max_parallel = INT_MIN;
    start = high_resolution_clock::now();
    #pragma omp parallel for reduction(max:max_parallel)
    for (int i = 0; i < array.size(); ++i) {
        if (array[i] > max_parallel) {
            max_parallel = array[i];
        }
    }
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);
    cout << "Maximum Element (Parallel): " << max_parallel << endl;
    cout << "Time Taken: " << duration.count() << " microseconds\n\n";
}

void calculateSum(const vector<int>& array) {
    int sum = 0;
    auto start = high_resolution_clock::now();
    for (int i = 0; i < array.size(); ++i) {
        sum += array[i];
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "Sum: " << sum << endl;
    cout << "Time Taken: " << duration.count() << " microseconds\n";

    int sum_parallel = 0;
    start = high_resolution_clock::now();
    #pragma omp parallel for reduction(+:sum_parallel)
    for (int i = 0; i < array.size(); ++i) {
        sum_parallel += array[i];
    }
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);
    cout << "Sum (Parallel): " << sum_parallel << endl;
    cout << "Time Taken: " << duration.count() << " microseconds\n\n";
}

void calculateAverage(const vector<int>& array) {
    double avg = 0.0;
    auto start = high_resolution_clock::now();
    for (int i = 0; i < array.size(); ++i) {
        avg += array[i];
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "Average: " << avg / array.size() << endl;
    cout << "Time Taken: " << duration.count() << " microseconds\n";

    double avg_parallel = 0.0;
    start = high_resolution_clock::now();
    #pragma omp parallel for reduction(+:avg_parallel)
    for (int i = 0; i < array.size(); ++i) {
        avg_parallel += array[i];
    }
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start);
    cout << "Average (Parallel): " << avg_parallel / array.size() << endl;
    cout << "Time Taken: " << duration.count() << " microseconds\n\n";
}

int main() {
    int N;
    const int MAX = 1000;
    cout << "Enter number of elements in array: ";
    cin >> N;

    vector<int> array(N);
    for (int i = 0; i < N; ++i) {
        array[i] = rand() % MAX;
    }

    findMinimum(array);
    findMaximum(array);
    calculateSum(array);
    calculateAverage(array);

    return 0;
}