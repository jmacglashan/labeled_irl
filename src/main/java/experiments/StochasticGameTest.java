package experiments;

import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.stochasticgames.GameEpisode;
import burlap.behavior.stochasticgames.auxiliary.GameSequenceVisualizer;
import burlap.domain.stochasticgames.gridgame.GGVisualizer;
import burlap.domain.stochasticgames.gridgame.GridGame;
import burlap.mdp.auxiliary.common.NullTermination;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.ActionType;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.stochasticgames.JointAction;
import burlap.mdp.stochasticgames.SGDomain;
import burlap.mdp.stochasticgames.model.JointRewardFunction;
import burlap.mdp.stochasticgames.world.World;
import burlap.mdp.stochasticgames.world.WorldObserver;
import burlap.shell.visual.SGVisualExplorer;
import burlap.visualizer.Visualizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static burlap.domain.stochasticgames.gridgame.GridGame.*;

/**
 * Created by ngopalan on 10/2/16.
 */
public class StochasticGameTest {

    public static void main(String[] args) {
        GridGame gg = new GridGame();

        SGDomain d = (SGDomain)gg.generateDomain();

        State s = GridGame.getCorrdinationGameInitialState();

        List<ActionType> actionTypes = d.getActionTypes();

        List<Action> actionList = new ArrayList<>();

        for(ActionType at: actionTypes){
            List<Action> al = at.allApplicableActions(s);
            actionList.addAll(al);
        }

        JointRewardFunction jrf = new JointRewardFunction() {
            @Override
            public double[] reward(State state, JointAction jointAction, State state1) {
                return new double[0];
            }
        };

        TerminalFunction jtf = new NullTermination();

        World env = new World(d,jrf, jtf, s);
        GameEpisode ga = new GameEpisode(s);




        List<JointAction> jaList = new ArrayList<JointAction>();
        for(Action a1:actionList){
            for(Action a2:actionList) {
                JointAction ja = new JointAction();
                ja.addAction(a1);
                ja.addAction(a2);

                env.executeJointAction(ja);
                ga.transition(ja,env.getCurrentWorldState(), new double[]{0.,0.});
                System.out.println(ja.actionName());
            }
        }

        ga.write("target/test");

        Visualizer v = GGVisualizer.getVisualizer(9, 9);
        new GameSequenceVisualizer(v, d, Arrays.asList(ga));


        SGVisualExplorer exp = new SGVisualExplorer(d, v, s);

        exp.addKeyAction("w", 0, ACTION_NORTH, "");
        exp.addKeyAction("s", 0, ACTION_SOUTH, "");
        exp.addKeyAction("d", 0, ACTION_EAST, "");
        exp.addKeyAction("a", 0, ACTION_WEST, "");
        exp.addKeyAction("q", 0, ACTION_NOOP, "");

        exp.addKeyAction("i", 1, ACTION_NORTH, "");
        exp.addKeyAction("k", 1, ACTION_SOUTH, "");
        exp.addKeyAction("l", 1, ACTION_EAST, "");
        exp.addKeyAction("j", 1, ACTION_WEST, "");
        exp.addKeyAction("u", 1, ACTION_NOOP, "");

        exp.initGUI();


    }

}
