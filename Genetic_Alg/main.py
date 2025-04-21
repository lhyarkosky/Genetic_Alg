#%%
#import functions and define main
import functions as fn

def main():
    rooms, times, facilitators, activities= fn.get_rtf("input.csv")
    initial_population=[]
    for i in range(1000): # 500 initial schedules
        random_schedule=fn.create_random_schedule(rooms, times, facilitators, activities)
        initial_population.append(random_schedule)
    best_schedule=fn.gen_alg(initial_population,150) #after 100 generations, return best schedule
    return(best_schedule)

#%%
#run main
best_sched=main() 

#%%
#see details on the best schedule (how it got scored)

fn.fitness(best_sched, verbose=True)
#%%
#save best schedule as csv
fn.save_as_csv(best_sched,filename='output.csv')
# %%

