use crate::game::GameState;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct CFRNode {
    pub regret_sum: Vec<f32>,
    pub strategy_sum: Vec<f32>,
    pub num_actions: usize,
}

impl CFRNode {
    pub fn new(num_actions: usize) -> Self {
        CFRNode {
            regret_sum: vec![0.0; num_actions],
            strategy_sum: vec![0.0; num_actions],
            num_actions,
        }
    }

    pub fn get_strategy(&mut self, realization_weight: f32) -> Vec<f32> {
        let mut strategy = vec![0.0; self.num_actions];
        let mut normalizing_sum = 0.0;

        for i in 0..self.num_actions {
            strategy[i] = if self.regret_sum[i] > 0.0 {
                self.regret_sum[i]
            } else {
                0.0
            };
            normalizing_sum += strategy[i];
        }

        for i in 0..self.num_actions {
            if normalizing_sum > 0.0 {
                strategy[i] /= normalizing_sum;
            } else {
                strategy[i] = 1.0 / self.num_actions as f32;
            }
            self.strategy_sum[i] += realization_weight * strategy[i];
        }

        strategy
    }
    
    pub fn get_average_strategy(&self) -> Vec<f32> {
        let mut avg_strategy = vec![0.0; self.num_actions];
        let normalizing_sum: f32 = self.strategy_sum.iter().sum();
        
        for i in 0..self.num_actions {
            if normalizing_sum > 0.0 {
                avg_strategy[i] = self.strategy_sum[i] / normalizing_sum;
            } else {
                avg_strategy[i] = 1.0 / self.num_actions as f32;
            }
        }
        avg_strategy
    }
}

pub struct CFRTrainer;

impl CFRTrainer {
    pub fn train(n_dice_p1: u8, n_dice_p2: u8, iterations: usize) -> HashMap<String, CFRNode> {
        let mut nodes = HashMap::new();
        for _ in 0..iterations {
            let game = GameState::new(n_dice_p1, n_dice_p2);
            Self::cfr(game, 1.0, 1.0, &mut nodes);
        }
        nodes
    }

    fn cfr(game: GameState, p0_weight: f32, p1_weight: f32, nodes: &mut HashMap<String, CFRNode>) -> f32 {
        let player = game.current_player;
        let valid_actions = game.get_valid_actions();
        
        if valid_actions.is_empty() {
            return 0.0;
        }

        let info_set = game.get_information_set();
        
        let node = nodes.entry(info_set.clone())
            .or_insert_with(|| CFRNode::new(valid_actions.len()));
            
        let strategy = node.get_strategy(if player == 0 { p0_weight } else { p1_weight });
        
        let num_actions = valid_actions.len();
        let mut util = vec![0.0; num_actions];
        let mut node_util = 0.0;

        // Vanilla CFR: Explore ALL actions
        for (i, action) in valid_actions.iter().enumerate() {
            let mut next_game = game.clone();
            let is_terminal = next_game.apply_action(action.clone());

            if is_terminal {
                util[i] = next_game.get_payoff();
            } else {
                if player == 0 {
                    util[i] = -Self::cfr(next_game, p0_weight * strategy[i], p1_weight, nodes);
                } else {
                    util[i] = -Self::cfr(next_game, p0_weight, p1_weight * strategy[i], nodes);
                }
            }
            node_util += strategy[i] * util[i];
        }

        // Re-access node to update regrets (CFR+ with regret floor at 0)
        let node_ref = nodes.get_mut(&info_set).unwrap();
        
        for i in 0..num_actions {
            let regret = util[i] - node_util;
            let weighted_regret = if player == 0 {
                p1_weight * regret
            } else {
                p0_weight * regret
            };
            
            // CFR+: Floor cumulative regret at 0 for faster convergence
            node_ref.regret_sum[i] = (node_ref.regret_sum[i] + weighted_regret).max(0.0);
        }

        node_util
    }
}
