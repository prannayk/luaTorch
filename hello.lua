function fact(n)
	if n < 2 then
		return 1
	else 
		return n * fact(n-1)
    end
end

function run()
    i=1; sum = 0
    while i <= 5 do
        sum = sum + (2*i - 1)
        i = i + 1
    end
    print(sum)
    local j = 1
    while j do
        print (j)
        j = j + 1
        if j >= 5 then
            break
        end
    end
    local max = function (a,b)
        return (a > b) and a or b
    end
    print(max(2,3))
end

function swap(a,b)
    return b,a
end

function printf(fmt, ...)
    io.write(string.format(fmt, ...))
end

function arrays()
    local a = {}
    for i = 1,6 do
        a[i] = math.random(20)
    end
    for i,x in ipairs(a) do
        print(x)
    end
    local set = {[math.random(10)] = true, [math.random(10)] = true}
    for i,x in pairs(set) do 
        print(tostring(i) .. "=" .. tostring(x))
    end
end

function iterators()
    function fromto(a,b)
        return function()
            if a > b then
                return nil
            else
                a = a + 1
                return a - 1
            end
        end
    end
    for i in fromto(2,5) do
        print(i)
    end
end
function fromtoo(a,b)
    return function(state)
        if state[1] >  state[2] then
            return nil
        else 
            state[1] = state[1] + 1
            return state[1] - 1
        end
     end, {a,b}
end

