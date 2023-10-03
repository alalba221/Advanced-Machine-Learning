p_list = [10,100,1000,2000];

n = 100;


timer1=[];
timer2=[];


for p = p_list
    A = rand(n,p);
    Ip = eye(p);
    In = eye(n);

    left = @()Left(A,Ip);
    right = @()Right(A,In);
    %timeit(left)
    timer1 = [timer1;timeit(left)];
    timer2 = [timer2;timeit(right)];
end


x = p_list;
y1 = timer1;
plot(x,y1,'DisplayName','Left')
title('Combine Plots')

hold on

y2 = timer2;
plot(x,y2,'DisplayName','Right')

xlim([0 2000])
ylim([-0.04 0.08])
hold off
legend
function Left(A,Ip)
    lambda = 1.0;
    inv((A'*A + Ip.*lambda))*A';
end

function Right(A,In)
    lambda = 1.0;
    A'*(inv(A*A' + In.*lambda));
end
