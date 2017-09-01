1;
clear all;

% PARAMETERS
global NUM_AGENTS = 3;
global NUM_TSTEPS = 49;
global NUM_TRUST_RESOLUTION = 199;

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
global rho = linspace(0.0001,0.9999,NUM_TRUST_RESOLUTION);



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
function e = expectation(pdf)
    global rho NUM_TRUST_RESOLUTION;
    % Check that @pdf is a row vector
    if (~isvector(pdf) || size(pdf) ~= [1,NUM_TRUST_RESOLUTION])
        error('PDF Argument not valid. Must be size 1xNUM_TRUST_RESOLUTION')
    end
    pdf = pdf .* rho;
    e = trapz(rho,pdf);
    % Sometimes the approx can be out of bounds, so better be save
    if (e > 1) e = 1;end
    if (e < 0) e = 0;end
end

function plotpdf(pdf, color)
    global rho NUM_TRUST_RESOLUTION;
    plot(rho,pdf,'Color',color);
    e = expectation(pdf)
    plot([e e], ylim, color);
end

function setup()
    global agents trusts NUM_AGENTS NUM_TRUST_RESOLUTION rho; 

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % SETUP AGENTS
    % (agent, attribute, time)
    % Set the agents attributes: 
    % 1 = Activity, 2 = Aptitude, 3 = Credence (in p)

    % Starting Credences are uniformly distributed  
    agents = rand(NUM_AGENTS, 3, 1);
    
    % For this simple model, there is just one Expert
    % AGENT 1 = EXPERT
    % AGENT 2 = TOTAL EVIDENCE STRATEGY
    % AGENT 3 = PREEMPTION STRATEGY
    agents(1,:,1) = [0.8 0.9 0.9];

    agents(2:end,1,1) = 0.5; 
    agents(2:end,2,1) = 0.5;
    agents(2:end,3,1) = 0.5;
    agents

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % SETUP TRUST FUNCTIONS
    % trusts(i,j,:): Trust function of agent i to agent j 
    trusts = zeros(NUM_AGENTS,NUM_AGENTS,NUM_TRUST_RESOLUTION);
    % The Expert knows herself to be an expert
    trusts(1,1,:) = betapdf(rho,5,0.4);

    for i = 2:NUM_AGENTS 
        % The Agents trust the expert already pretty much
        trusts(i,1,:) = betapdf(rho,5,1.2);
        % The Agents are slightly overconfident
        trusts(i,i,:) = betapdf(rho,5,4);
        % The expert doesnt think the other agents to be very helpful
        trusts(1,i,:) = betapdf(rho,5,5);
    end

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % SETUP NETWORK
    % network(i,j): There is a link from agent i to agent j
    % i 'listens to' j
    network = zeros(NUM_AGENTS, NUM_AGENTS);
    network(2,1) = 1;
    network(3,1) = 1;
end

% One Complete timestep
% Every Timestep, the following happens:
% 1) with a chance, the expert inquires
%   If inquired
%   1.1) Updates Credence
%   1.2) Updates Trust Function
%   1.3) if inquired, and if sufficiently confident, presents new findings to agents
% (This can be stupid, since if the expert is less condident than before but still confident enough, the agent will get more confident which is the opposite of the intended effect. Needs examination).
% 2) With a chance, agents inquire themselves.
% 3) Total Evidence Agents combine their sources to update their opinion
% 4) Preemption Agents will adopt experts opinion if they regard them as an authority, otherwise like TEA.
% 5) Trust functions get updated
function step()
    global rho t agents trusts NUM_AGENTS;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % 1) Expert Inquiry
    if (coin(agents(1,1,1)))

    end

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
            % cf. Angere(ms), p.4/5
            c = agents(i,3,t);
            tau = expectation(squeeze(trusts(i,i,:))'); 
            nc = 1 - c ;
            ntau = 1 - tau;
            if (s == 1)
                agents(i,3,t+1) = c*tau / (c*tau + nc*ntau);
            else
                agents(i,3,t+1) = c*ntau / (c*ntau + nc*tau); 
            end

            % Calculate new Trust function
            % cf. Angere(ms), p.8/9
            c = agents(i,3,t);
            exptau = expectation(squeeze(trusts(i,i,:))');
            nc = 1 - c;
            nexptau = 1 - exptau;

            tau = squeeze(trusts(i,i,:))';
            if (s == 1)
                dnom = exptau*c + nexptau*nc;
                trusts(i,i,:) = tau .* (((rho*c) .+ ((1-rho)*nc)) / dnom);
            else
                dnom = exptau * nc + nexptau * c;
                trusts(i,i,:) = tau .* (((rho*nc) .+ ((1-rho)*c)) / dnom);
            end
        end
    end

    t++;
end

function main()
    global NUM_TSTEPS NUM_AGENTS agents trusts;
    setup();
    %for i = 1:99
        %step();
        %hold on
    %plotpdf(squeeze(trusts(1,1,:))', [i/100,0,0]);

    %end
end

main();

