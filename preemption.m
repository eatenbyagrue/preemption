1;
clear all;

% PARAMETERS
global NUM_AGENTS = 2;
global NUM_TSTEPS = 49;
global NUM_TRUST_RESOLUTION = 99;

% In my 'elaborate' model, the target proposition is true
global p = true;
% 3dim Matrix to hold most agent's properties, third dimension is time
global agents = [];
% 3dim Matrix to hold all trust levels between agents, 3rd dimension is reliability unit interval 
global trusts = [];
% Keeps track of the time
% We start with t = 1 b/c octave starts arrays with 1.
% Less confusing this way.
global t = 1;

% Holds all values that the reliability can take 
% Used as X-Vector for PDFs
global rho = linspace(0,1,NUM_TRUST_RESOLUTION);



% Simple function that returns TRUE with probabiliy @bias.
function result = coin(bias)
    result = binornd(1,bias,1);
end

% No ternary operator makes me go crazy
function retval = ternary (expr, trueval, falseval)
    if (expr)
        retval = trueval;
    else
        retval = falseval;
    end
end

% Calculates and returns the Expectation (Mean, Expected Value)
% of @pdf over [0,1] with resolution NUM_TRUST_RESOLUTION
% by approximating the integral. Crank up resolution to get more precise.
function expectation = calcExpectation(pdf)
    global rho NUM_TRUST_RESOLUTION;
    % Check that @pdf is a row vector
    if (~isvector(pdf) || size(pdf) ~= [1,NUM_TRUST_RESOLUTION])
        error('PDF Argument not valid. Must be size 1xNUM_TRUST_RESOLUTION')
    end
    pdf = pdf .* rho;
    expectation = trapz(rho,pdf);
end

function drawpdf(pdf, linespec)
    global rho NUM_TRUST_RESOLUTION;
    plot(rho,pdf,linespec);
    e = calcExpectation(pdf);
    plot([e e], ylim, linespec);
end

function setup()
    global agents trusts NUM_AGENTS NUM_TRUST_RESOLUTION rho; 

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % SETUP AGENTS
    % (agent, attribute, time)
    % Set the agents attributes: 
    % 1 = Activity, 2 = Aptitude, 3 = Credence (in p)
    % Starting Credences are Randomized
    agents = rand(NUM_AGENTS, 3, 1);
    agents(:,1,1) = 1; 
    agents(:,2,1) = 1;
    agents(:,3,1) = 0.8;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % SETUP TRUST FUNCTIONS
    % trusts(i,j,:): Trust function of agent i to agent j 

    trusts = zeros(NUM_AGENTS,NUM_AGENTS,NUM_TRUST_RESOLUTION);
    trusts(1,1,:) = betapdf(rho,5,4.5);
    trusts(1,2,:) = betapdf(rho,5,3);
    trusts(2,1,:) = betapdf(rho,2,5);
    trusts(2,2,:) = betapdf(rho,5,2);

    %hold on

    %tr =squeeze(trusts(1,1,:));
    %plot(rho,tr,'-r');
    %e = calcExpectation(squeeze(tr)')
    %plot([e e], ylim, '--r');

    %tr =trusts(1,2,:);
    %plot(rho,tr,'-g');
    %e = calcExpectation(squeeze(tr)')
    %plot([e e], ylim, '--g');


    %tr =trusts(2,1,:);
    %plot(rho,tr,'-b');
    %e = calcExpectation(squeeze(tr)')
    %plot([e e], ylim, '--b');


    %tr =trusts(2,2,:);
    %plot(rho,tr,'-k');
    %e = calcExpectation(squeeze(tr)')
    %plot([e e], ylim, '--k');

    %hold off
end

% One Complete timestep
function step()
    global rho t agents trusts NUM_AGENTS;

    for i = 1:NUM_AGENTS
        % Activity and Aptitude are constant over time 
        activity = agents(i,1,1);
        aptitude = agents(i,2,1);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        % SOURCE : INQUIRY       %
        if (coin(activity))
            % Inquiry based on Aptitude
            s = coin(aptitude);

            % Calculate new Credence from Prior, Trust, and their inverses
            c = agents(i,3,t);
            tau = calcExpectation(squeeze(trusts(i,i,:))'); 
            nc = 1 - c ;
            ntau = 1 - tau;
            if (s == 1)
                agents(i,3,t+1) = c*tau / (c*tau + nc*ntau);
            else
                agents(i,3,t+1) = c*ntau / (c*ntau + nc*tau); 
            end

            % Calculate new Trust function
            c = agents(i,3,t);
            tau = calcExpectation(squeeze(trusts(i,i,:))');
            nc = 1 - c;
            ntau = 1 - tau;

            tauDummy = squeeze(trusts(i,i,:))';
            %if (s == 1)
            nom = (rho*c) .+ ((1-rho)*nc)
            dnom = tau*c + ntau*nc
            foo =  nom / dnom
            newtau = tauDummy .* foo;
            trusts(i,i,:) = newtau;
            %end
        end
    end

    t++;
end

function main()
    global NUM_TSTEPS NUM_AGENTS agents trusts;
    setup();
    hold on
    drawpdf(squeeze(trusts(1,1,:))','-r');
    step();
    drawpdf(squeeze(trusts(1,1,:))','-g');
    %for i = 1:NUM_TSTEPS
    %
    %step();
    %end
    %hold on
    %for i = 1:NUM_AGENTS
    %plot(squeeze(agents(i,3,:)));
    %end
    %hold off
end

main();

%hold on
%for i = 1:NUM_AGENTS
%credences = squeeze(agents(i,3,:))
%length(credences)
%length(0:NUM_TSTEPS)
%length(1:NUM_TSTEPS)
%plot(0:NUM_TSTEPS,credences);
%end
%pause
%hold off
