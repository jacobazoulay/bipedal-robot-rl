#plotmaker.jl
#I'm doing this in julia because it's easier than in Python
using PGFPlots, DataFrames, CSV, RollingFunctions, LinearAlgebra

df = DataFrame(CSV.File("data/dqn.csv"))
name_array = names(df)
# temp = Matrix(df)
# samples = I*temp'
rollouts = Matrix(df)
episodes = (rollouts[:,1]*I)[:]
num_episodes = size(episodes)[1]
reward = (rollouts[:,4]*I)[:] #col 4 for dqn
# reward_std = (rollouts[:,5]*I)[:]
# threshold =
moving_avg = 50
smoothed_reward = rolling(sum, reward, moving_avg)./moving_avg

p = Axis([
    Plots.Scatter(episodes, reward, legendentry="Rollout Data", markSize=0.1),
    Plots.Linear(episodes[moving_avg:end], smoothed_reward, legendentry="Average Reward", mark="none"),
    Plots.Linear(episodes, 300*ones(num_episodes), legendentry="Reward Threshold", mark="none", style="black,dashed")
    ], xlabel="Episode", ylabel="Reward", title="Accumulated Reward vs Episode", xmin=0, xmax=num_episodes
    )
# savefig("plots/biped_ars001.pdf")
# a = Axis(Plots.Linear(x, y, legendentry="My Plot"), xlabel="X", ylabel="Y", title="My Title")
# p = Axis(Plots.Linear(episodes, rollouts, legendentry="reward"))#, markSize=0.1, onlyMarks=true)
# p = Axis(Plots.Linear(episodes, rollouts, legendentry="reward", markSize=0.1, onlyMarks=true), xlabel="Episode", ylabel="Reward", title="Episode Reward vs Episode Number")
p.legendStyle = "at={(1.05,1.0)}, anchor=north west"
save("plots/dqn.pdf", p)