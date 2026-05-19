#include "stdio.h"

void merge(int *arr, int left, int mid, int right) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int L[n1], R[n2];

    for (i = 0; i < n1; i++) {
        L[i] = arr[left + i];
    }
    for (j = 0; j < n2; j++) {
        R[j] = arr[mid + j + 1];
    }

    i = 0; j = 0;
    k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergesort(int *arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergesort(arr, left, mid);
        mergesort(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

// Hàm hỗ trợ in mảng
void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

int main() {
    // Test Case 1: Mảng ngẫu nhiên cơ bản
    int arr1[] = {12, 11, 13, 5, 6, 7};
    int n1 = sizeof(arr1) / sizeof(arr1[0]);
    printf("Test 1 - Mang ban dau: ");
    printArray(arr1, n1);
    mergesort(arr1, 0, n1 - 1);
    printf("Test 1 - Sau khi sap xep: ");
    printArray(arr1, n1);
    printf("---------------------------\n");

    // Test Case 2: Mảng có số âm và số trùng
    int arr2[] = {3, -2, 1, 3, 5, 1, -8, 9};
    int n2 = sizeof(arr2) / sizeof(arr2[0]);
    printf("Test 2 - Mang ban dau: ");
    printArray(arr2, n2);
    mergesort(arr2, 0, n2 - 1);
    printf("Test 2 - Sau khi sap xep: ");
    printArray(arr2, n2);

    return 0;
}