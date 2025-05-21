using JuMP
# using GLPK
using CPLEX

using DataFrames
using Dates
using DataStructures
using CSV
using MathOptInterface



function load_data(path)
    start = now()
    input_data = Dict()
    result_data = Dict()
    centerDemand = CSV.read(path*"centerDemand.csv", DataFrame)
    specialistHourlyCapacity = CSV.read(path*"specialistHourlyCapacity.csv", DataFrame)
    specialistAvailability = CSV.read(path*"specialistAvailability.csv", DataFrame)
    specialistPreference = CSV.read(path*"specialistPreference.csv", DataFrame)
    centerMaxBacklogSize = CSV.read(path*"centerMaxBacklogSize.csv", DataFrame)
    centerFullTimeCoverPerc = CSV.read(path*"centerFullTimeCoverPerc.csv", DataFrame)
    hourIsMixed = CSV.read(path*"hourIsMixed.csv", DataFrame)


    # get basic set
    D = unique(centerDemand.d)
    H = unique(centerDemand.h)
    C = unique(centerDemand.c)
    S = unique(specialistPreference.s)

    # get compound set for valid combination
    DHS1 = [(row.d, row.h, row.s) for row in eachrow(specialistAvailability)]
    SC = [(row.s, row.c) for row in eachrow(specialistPreference)]

    DHS = [(d, h, s) for (d, h, s) in DHS1 for ss in S if ss == s]

    # turn in to dictionary
    input_data["centerDemand"] = Dict(
        (row.d, row.h, row.c) => row.CenterDemand for row in eachrow(centerDemand)
    )
    input_data["specialistPreference"] = Dict(
        (row.s, row.c) => row.SpecialistPreference for row in eachrow(specialistPreference)
    )
    specialistHourlyCapacity = filter(row -> row.s in S, specialistHourlyCapacity)
    input_data["specialistHourlyCapacity"] = Dict(
        row.s => row.SpecialistHourlyCapacity for row in eachrow(specialistHourlyCapacity)
    )
    input_data["specialistIsFullTime"] = Dict(
        row.s => row.SpecialistIsFullTime for row in eachrow(specialistHourlyCapacity))
    input_data["specialistAvailability"] = Dict(
        (row.d, row.h, row.s) => row.SpecialistAvailability for row in eachrow(specialistAvailability)
    )
    input_data["centerMaxBacklogSize"] = Dict(
        (row.d, row.h, row.c) => row.CenterMaxBacklogSize for row in eachrow(centerMaxBacklogSize)
    )
    input_data["centerFullTimeCoverPerc"] = Dict(
        row.c => row.CenterFullTimeCoverPerc for row in eachrow(centerFullTimeCoverPerc)
    )
    input_data["hourIsMixed"] = Dict(
        row.h => row.HourIsMixed for row in eachrow(hourIsMixed)
    )

    specialistAtCenterByHour = CSV.read(path*"specialistAtCenterByHour.csv", DataFrame)
    specialistAtCenter = CSV.read(path*"specialistAtCenter.csv", DataFrame)
    specialistWorks = CSV.read(path*"specialistWorks.csv", DataFrame)

    result_data["specialistAtCenterByHour"] = Dict(
     (row.d, row.h, row.s, row.c) => row.specialistAtCenterByHour for row in eachrow(specialistAtCenterByHour)
    )
    result_data["specialistAtCenter"] = Dict(
     (row.s, row.c) => row.specialistAtCenter for row in eachrow(specialistAtCenter)
    )
    result_data["specialistWorks"] = Dict(
     (row.s) => row.specialistWorks for row in eachrow(specialistWorks)
    )
    endt = now()
    println("$(now()) finish reading $(endt - start)")
    return D, H, C, S, DHS, SC, input_data,result_data
end


function solve_assignment(D, H, C, S, DHS, SC, input_data,result_data;  is_solve=true,lp_solve=false)
    if lp_solve
        println(now()," Starting lp model")
    else
        println(now()," Starting mip model")
    end

    model_start_time = now()

    model = Model(CPLEX.Optimizer)
#     model.Model(GLPK.Optimizer)

    # create necessary subset for model forumlation
    DHSC = [(d, h, s, c) for (d, h, s) in DHS for (ss, c) in SC if ss == s]
    DHS_C = Dict(dhs => Any[] for dhs in DHS)
    for dhsc in DHSC
        push!(DHS_C[(dhsc[1],dhsc[2],dhsc[3])], dhsc)
    end

    DHC = [(d, h, c) for d in D, h in H, c in C]
    DS = unique([(d, s) for (d, h, s, c) in DHSC])
    SS = unique([s for (d,s) in DS])

    mixedC = [c for (c,fullPerc) in input_data["centerFullTimeCoverPerc"]]
    mixedH = [h for (h, is_mixed) in input_data["hourIsMixed"] if is_mixed == 1]
    FulltimeS = [s for (s, is_fulltime) in input_data["specialistIsFullTime"] if coalesce(is_fulltime, 0) == 1]

    # define parameters
    centerDemand = Dict((d, h, c) => get(input_data["centerDemand"], (d, h, c), 0) for d in D for h in H for c in C)
    specialistHourlyCapacity = Dict(s => input_data["specialistHourlyCapacity"][s] for s in S)
    specialistPreference = Dict((s, c) => input_data["specialistPreference"][(s, c)] for (s, c) in SC)
    centerMaxBacklogSize = Dict((d, h, c)  => get(input_data["centerMaxBacklogSize"] ,(d, h, c), 0) for d in D for h in H for c in C)
    centerFullTimeCoverPerc = Dict((c) => get(input_data["centerFullTimeCoverPerc"], c, 0) for c in mixedC )
    webClockRatio = 0.8

    model_start_time = now()
    # define variables
    @variable(model, specialistWorkloads[dhsc in DHSC] >= 0)
    @variable(model, centerHourlyUnmetDemand[dhc in DHC] >= 0)

    # Create dictionary before constraint building and pre-allocate memory for better performance.
    DHC_to_DHSC = Dict{Tuple{Int,Int,Int}, Vector{Tuple{Int,Int,Int,Int}}}(dhc => [] for dhc in DHC)
    sizehint!(DHC_to_DHSC, length(DHC))
    for dhsc in DHSC
        dhc = (dhsc[1], dhsc[2], dhsc[4])
        if dhc in DHC
            push!(DHC_to_DHSC[dhc], dhsc)
        end
    end

    if lp_solve
        webClockRatio = 0.9

        @variable(model, 0<= centerHourlyBacklog[dhc in DHC] <= centerMaxBacklogSize[dhc])
        @variable(model, centerFullTimeUnmetDemand[d in D, c in mixedC] >= 0)
        @objective(model, Min, sum(centerHourlyUnmetDemand[dhc] * 100 for dhc in DHC)+
                    sum(centerFullTimeUnmetDemand[d ,c] * 10 for d in D for c in mixedC) +
                    sum(centerHourlyBacklog[dhc] for  dhc in DHC))

        specialistAtCenterByHour_v = Dict((d, h, s, c) => val for ((d, h, s, c), val) in result_data["specialistAtCenterByHour"])
        @constraint(model, specialistWorkloadsCon[dhsc in DHSC], specialistWorkloads[dhsc] <= specialistHourlyCapacity[dhsc[3]] * webClockRatio * get(specialistAtCenterByHour_v,dhsc,0))
        @constraint(model,centerDemandBalanceCon[dhc in DHC],
            sum(specialistWorkloads[dhsc] for dhsc in DHC_to_DHSC[dhc]) +
            centerHourlyBacklog[dhc] +
            centerHourlyUnmetDemand[dhc] ==
            centerDemand[dhc] +
            centerHourlyBacklog[
               dhc[2] == 0 ?
                (dhc[1] == 1 ? (length(D), length(H)-1, dhc[3]) : (dhc[1] - 1, length(H)-1, dhc[3])) :
                (dhc[1], dhc[2] - 1, dhc[3])
            ])

        @constraint(model, fullTimeCoverageCon[d in D, c in mixedC],
        sum(
            specialistWorkloads[dhsc]
            for dhsc in DHSC
            if dhsc[1] == d && dhsc[2] in mixedH && dhsc[3] in FulltimeS && dhsc[4] == c
        )
        >= centerFullTimeCoverPerc[c]/100 * sum(centerDemand[dhc] for dhc in DHC if dhc[1]==d && dhc[2] in mixedH && dhc[3]==c)
           - centerFullTimeUnmetDemand[d, c]
        )

    else
        @variable(model, specialistWorks[s in S], Bin)
        @variable(model, specialistAtCenter[sc in SC], Bin)
        @variable(model, specialistAtCenterByHour[dhsc in DHSC], Bin)

        # define objective
        @objective(model, Min, sum(centerHourlyUnmetDemand[dhc] for dhc in DHC) + sum(specialistAtCenter[sc] *specialistPreference[sc]  for sc in SC))

        # define constraint
        @constraint(model, specialistAtCenterCon[sc in SC ], specialistAtCenter[sc] <= specialistWorks[sc[1]])
        @constraint(model, specialistAtCenterByHourCon[dhsc in DHSC], specialistAtCenterByHour[dhsc] <= specialistAtCenter[(dhsc[3], dhsc[4])])
        @constraint(model, specialistOneCenterPerHourCon[dhs in DHS], sum(specialistAtCenterByHour[i] for i in DHS_C[dhs]) <= 1)

        @constraint(model, specialistWorkloadsCon[dhsc in DHSC], specialistWorkloads[dhsc] <= specialistHourlyCapacity[dhsc[3]] * webClockRatio * specialistAtCenterByHour[dhsc])
        @constraint(model, centerDemandBalanceCon[dhc in DHC],
            sum(specialistWorkloads[dhsc] for dhsc in DHC_to_DHSC[dhc]) +
            centerHourlyUnmetDemand[dhc] == centerDemand[dhc])

    end

    # print model size
    model_end_time = now()
    println("Model generation time: ", model_end_time - model_start_time)

    num_variables = length(all_variables(model))
    println("Number of variables: $num_variables")

    num_constraints = length(all_constraints(model,include_variable_in_set_constraints=false))
    println("Number of constraints: $num_constraints")

    # solve model
    if is_solve
        set_optimizer_attribute(model, "CPXPARAM_ScreenOutput", 0)
        start = now()
        println(start," Start solving")
        set_optimizer_attribute(model, "CPX_PARAM_EPGAP", 0.05)
        optimize!(model)
        println("Objective value: ", objective_value(model))
        endt = now()
        println("$(now()) finish solving $(endt - start)")
    end

    return (model=model, centerDemandBalanceCon=centerDemandBalanceCon)

end

data_path="Data\\"
D, H, C, S, DHS, SC, input_data,result_data = load_data(data_path)
instance_mip = solve_assignment(D, H, C, S, DHS, SC, input_data, result_data, is_solve=true)
instance_lp = solve_assignment(D, H, C, S, DHS, SC, input_data, result_data, is_solve=true, lp_solve=true)

# update demand
println("Model update start: ", now())
start_time = now()
ms=0.0
for i in 1:10
    scale = 0.7 + i * 0.05
    newCenterDemand = Dict(k => v * scale for (k, v) in input_data["centerDemand"])
    tmp1=now()
    for j in keys(newCenterDemand)
     set_normalized_rhs(instance_lp.centerDemandBalanceCon[j], newCenterDemand[j])
    end
    tmp2 = Millisecond(now()-tmp1).value / 1000.0
    global ms += tmp2
    optimize!(instance_lp.model)

end
end_time = now()
println("Model update end: ", now())
println("Model update time: ", ms)
println("Model update total time: ", end_time - start_time)
