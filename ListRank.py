from mpi4py import MPI

def list_ranking(rank, size, local_list, full_list):
    local_rankings = [0] * len(local_list)
    for i in range(len(local_list)):
        local_rankings[i] = sum(1 for j in range(len(local_list)) if j != i and local_list[j] < local_list[i])
    all_rankings = comm.gather(local_rankings)  
    if rank == 0:
        flattened_rankings = [item for sublist in all_rankings for item in sublist]
        
        print("Lista Completa sin Ordenar:", full_list)
        
        sorted_indices = sorted(range(len(flattened_rankings)), key=lambda k: flattened_rankings[k])
        sorted_data = [full_list[i] for i in sorted_indices]

        print("Lista Completa Ordenada:", sorted_data)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    full_list = [10, 7, 5, 8, 3, 1, 6, 9, 4, 2]
    local_list = full_list[rank::size]
    list_ranking(rank, size, local_list, full_list)
