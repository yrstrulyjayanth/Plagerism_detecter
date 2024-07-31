package edu.northeastern.cs5500.strategies;

import edu.northeastern.cs5500.strategies.implementations.LCS;
import edu.northeastern.cs5500.strategies.implementations.LevenshteinDistance;
import edu.northeastern.cs5500.strategies.implementations.WeightedScore;
import edu.northeastern.cs5500.strategies.implementations.ast.lcs.LongestCommonSubSequence;
import edu.northeastern.cs5500.strategies.implementations.ast.treeeditdistance.AstTreeEditDistance;
import edu.northeastern.cs5500.strategies.implementations.moss.MossComparison;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;


/**
 * @author Praveen Singh, namratabilurkar
 */
@RunWith(SpringRunner.class)
@SpringBootTest
public class StrategyFactoryTest{

    @Autowired
    LevenshteinDistance levenshteinDistance;

    @Autowired
    LCS lcs;
    
    @Autowired
    LongestCommonSubSequence longestCommonSubSequence;

    @Autowired
    AstTreeEditDistance astTreeEditDistance;

    @Autowired
    StrategyFactory strategyFactory;

    @Autowired
    WeightedScore weightedScore;

    @Autowired
    MossComparison mossComparison;


    /**
     * test for getting levenshtein distance strategy when it is provided
     */
    @Test
    public void getLevenshteinDistanceStrategyShouldReturnTheExpectedStrategy(){
        Assert.assertEquals(levenshteinDistance,
                strategyFactory.getStrategyByStrategyType("LEVENSHTEIN_DISTANCE"));
    }
    
    /**
     * test for getting lcs strategy when it is provided
     */
    @Test
    public void getLCSStrategyShouldReturnTheExpectedStrategy(){
        Assert.assertEquals(lcs,
                strategyFactory.getStrategyByStrategyType("LCS"));
    }
    
    /**
     * test for getting AST lcs strategy when it is provided
     */
    @Test
    public void getASTLcsStrategyShouldReturnTheExpectedStrategy(){

                strategyFactory.getStrategyByStrategyType("AST_LCS");
    }

    /**
     * test for getting AST tree edit distance strategy when it is provided
     */
    @Test
    public void getASTTreeEditStrategyShouldReturnTheExpectedStrategy(){

        strategyFactory.getStrategyByStrategyType("AST_TREE_EDIT_DISTANCE");
    }

    @Test
    public void getMosShouldReturnTheExpectedStrategy(){

        strategyFactory.getStrategyByStrategyType("MOS");
    }

    /**
     * test for getting Weighted score strategy when it is provided
     */
    @Test
    public void getWeightedScoreStrategyShouldReturnTheExpectedStrategy(){
        Assert.assertEquals(weightedScore,
                strategyFactory.getStrategyByStrategyType("WEIGHTED_SCORE"));
    }

    /**
     * test for getting Levenshtein distance strategy when invalid strategy is provided
     */
    @Test
    public void getLevenshteinDistanceStrategyShouldReturnTheDefaultStrategyIfInvalidStrategyProvided(){
        Assert.assertEquals(levenshteinDistance,
                strategyFactory.getStrategyByStrategyType("INVALID"));
    }

    /**
     * test for getting Levenshtein distance strategy when null is provided
     */
    @Test
    public void getLevenshteinDistanceStrategyShouldReturnDefaultStrategyWhenNullProvided(){
        Assert.assertEquals(levenshteinDistance, strategyFactory.getStrategyByStrategyType(null));
    }

}
