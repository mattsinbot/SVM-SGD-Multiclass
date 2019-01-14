clc
clear
close all


% Load training and validation data
% Warning: make sure VL_FEAT is installed
run('/home/nobug-ros/Documents/vlfeat/toolbox/vl_setup')
load('../data/q3_2_data.mat');


%change all -1 to 2m
trLb ( trLb==-1  ) = 2;
valLb( valLb==-1 ) = 2;


%=================
% tunable params
%=================
epoch    = 2000;
eta_0    = 1;
eta_1    = 100;
c_factor = 0.7;

%=======================
% paramaters from data
%=======================
[ dimensions, n_tr  ] = size( trD );            % dimensions, train_data_size
[ classes   , dummy ] = size( unique(trLb) );   % calculate unique classes
w = zeros(dimensions, classes);                 % initialise w

%===================
% loop starts here
%===================
% this loop is for epochs
cumulative_loss = 0;
losses = zeros(epoch, 2);
c_mul_trD = c_factor*trD;
for epoch_no = 1:100
    disp(epoch_no)
    eta = eta_0/(eta_1 + epoch_no);
    permutation = randperm(n_tr);

    % this loop is for permutated xi's
    loss_temp = zeros(n_tr,1);
    for iter = permutation
        xi      = trD(:,iter);
        yi      = trLb(iter);
        yi_hat  = w' * xi;
        [ B, I] = maxk(yi_hat, 2);        
        
        if ( I(1)==yi )
            yi_hat=I(2); 
        else
            yi_hat=I(1);
        end

        loss = max( w(:,yi_hat)'*xi - w(:,yi)'*xi +1, 0);
        w_by_n = w/n_tr;
        % this loop is for all classes
        class_temp = zeros(classes, 1);
        for curr_class = 1:classes
            delta = w_by_n(:,curr_class);
            if (loss>0) && ( curr_class==yi || curr_class==yi_hat)
                sign = 1;
                if(curr_class==yi); sign = -1; end
                delta = delta + sign*c_mul_trD(:,iter);
            end
            w(:,curr_class) = w(:,curr_class) - eta*delta;
            class_temp(curr_class, 1) = w(:,curr_class)'*w(:,curr_class);
        end
        
        loss_temp(iter,1) = sum(class_temp)/(2*n_tr) + c_factor*max( w(:,yi_hat)'*xi - w(:,yi)'*xi +1, 0);

    end
    losses(epoch_no, 2) = sum(loss_temp);
    losses(epoch_no, 1) = epoch_no;
end

disp("training done.!")

get_accuracy(trD, trLb, w, 0);
[accuracy, ~] = get_accuracy(valD, valLb, w, 0);

disp("objective");
disp( losses(epoch, 2) );

sum = 0;
for i=1:classes
    temp = w(:,i);
    total = temp'*temp;
    sum = sum + total;
end
disp("w-norm");
disp(sum);

figure
plot(losses(:,1), losses(:,2) )
title("c=10")
xlabel("epochs")
ylabel("loss")

function [accuracy, predicted] = get_accuracy(data, predictions, w, b)
    [ ~, n  ] = size( data );
    predicted = data'*w + b;
    [~, predicted] = max(predicted, [], 2);
    class_diff = predictions - predicted;
    incorrect = size(nonzeros(class_diff));
    correct = n - incorrect(1);
    accuracy = correct/n;
    disp("validation accuracy=");
    disp(accuracy);
end
