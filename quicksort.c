#include "stdlib.h"
#include "stdio.h"

void swap(int* a, int* b) {
    if (a != b) {
        *a = *a ^ *b;
        *b = *a ^ *b;
        *a = *a ^ *b;
    }
}

int partition(int *arr, int left, int right) {
    int pivot = arr[right];
    int i = left - 1;

    for (int j = left; j < right; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }

    swap(&arr[i + 1], &arr[right]);
    return i + 1;
}

void quicksort(int *arr, int left, int right) {
    if (left < right) {
        int pivot = partition(arr, left, right);
        quicksort(arr, left, pivot - 1);
        quicksort(arr, pivot + 1, right);
    }
}

void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    // Test Case 1: Mảng ngẫu nhiên cơ bản
    int arr1[] = {10, 7, 8, 9, 1, 5};
    int n1 = sizeof(arr1) / sizeof(arr1[0]);
    printf("Test 1 - Mang ban dau: ");
    printArray(arr1, n1);
    quicksort(arr1, 0, n1 - 1);
    printf("Test 1 - Sau khi sap xep: ");
    printArray(arr1, n1);
    printf("---------------------------\n");

    // Test Case 2: Mảng có các phần tử trùng lặp và số âm
    int arr2[] = {3, -2, 1, 3, 5, 1, -8, 9};
    int n2 = sizeof(arr2) / sizeof(arr2[0]);
    printf("Test 2 - Mang ban dau: ");
    printArray(arr2, n2);
    quicksort(arr2, 0, n2 - 1);
    printf("Test 2 - Sau khi sap xep: ");
    printArray(arr2, n2);
    printf("---------------------------\n");

    // Test Case 3: Mảng đã được sắp xếp sẵn (Test trường hợp xấu nhất của Quicksort)
    int arr3[] = {1, 2, 3, 4, 5, 6};
    int n3 = sizeof(arr3) / sizeof(arr3[0]);
    printf("Test 3 - Mang ban dau: ");
    printArray(arr3, n3);
    quicksort(arr3, 0, n3 - 1);
    printf("Test 3 - Sau khi sap xep: ");
    printArray(arr3, n3);

    return 0;
}