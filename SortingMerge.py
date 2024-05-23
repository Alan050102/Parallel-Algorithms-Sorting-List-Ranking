from mpi4py import MPI
import numpy as np

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result

def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) 
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def parallel_merge_sort(arr, comm):
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    if rank == 0:
        data = np.array_split(arr, size)
    else:
        data = None

    local_data = comm.scatter(data, root=0)
    local_sorted = merge_sort(local_data)

    sorted_data = comm.gather(local_sorted, root=0)
    
    if rank == 0:
        # Ordenar por metodo de merge sort
        while len(sorted_data) > 1:
            new_sorted = []
            for i in range(0, len(sorted_data), 2):
                if i + 1 < len(sorted_data):
                    new_sorted.append(merge(sorted_data[i], sorted_data[i + 1]))
                else:
                    new_sorted.append(sorted_data[i])
            sorted_data = new_sorted

        return sorted_data[0]
    return None
  
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # Definición del arreglo
    arr = np.array([6, 2, 9, 3, 8, 5, 1, 7, 4, 0])  
    if rank == 0:
        print(f"Arreglo original: {arr}")

    sorted_arr = parallel_merge_sort(arr, comm)

    if rank == 0:
        print(f"Arreglo ordenado: {sorted_arr}")
