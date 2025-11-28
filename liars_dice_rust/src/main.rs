mod game;
mod cfr;

use crate::cfr::{CFRTrainer, CFRNode};
use crate::game::{Action, GameState};
use rayon::prelude::*;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

fn action_to_str(action: &Action) -> String {
    match action {
        Action::Challenge => "Challenge".to_string(),
        Action::Bid(q, f) => format!("{}-{}", q, f),
    }
}

fn save_strategy(nodes: &HashMap<String, CFRNode>, n_dice_p1: u8, n_dice_p2: u8) {
    let filename = format!("../strategy_{}v{}.csv", n_dice_p1, n_dice_p2);
    println!("Saving strategy to {}...", filename);

    let mut file = File::create(filename).expect("Unable to create file");
    writeln!(file, "InfoSet,Action,Probability").expect("Unable to write header");

    for (info_set, node) in nodes {
        let avg_strategy = node.get_average_strategy();

        // Reconstruct actions
        let parts: Vec<&str> = info_set.split('|').collect();
        let bid_str = parts[1];
        
        let mut dummy_game = GameState::new(n_dice_p1, n_dice_p2);
        if bid_str != "None" {
            let b_parts: Vec<&str> = bid_str.split('-').collect();
            let q = b_parts[0].parse::<u8>().unwrap();
            let f = b_parts[1].parse::<u8>().unwrap();
            dummy_game.current_bid = Some((q, f));
        } else {
            dummy_game.current_bid = None;
        }

        let valid_actions = dummy_game.get_valid_actions();

        for (i, prob) in avg_strategy.iter().enumerate() {
            if *prob > 0.001 {
                let action_str = action_to_str(&valid_actions[i]);
                writeln!(file, "{},{},{}", info_set, action_str, prob).unwrap();
            }
        }
    }
    println!("Save complete.");
}

fn merge_nodes(mut map1: HashMap<String, CFRNode>, map2: HashMap<String, CFRNode>) -> HashMap<String, CFRNode> {
    for (key, node2) in map2 {
        let node1 = map1.entry(key).or_insert_with(|| CFRNode::new(node2.num_actions));
        
        for i in 0..node1.num_actions {
            node1.regret_sum[i] += node2.regret_sum[i];
            node1.strategy_sum[i] += node2.strategy_sum[i];
        }
    }
    map1
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        println!("Usage: cargo run <p1_dice> <p2_dice> <iterations>");
        return;
    }

    let p1_dice: u8 = args[1].parse().expect("Invalid p1 dice");
    let p2_dice: u8 = args[2].parse().expect("Invalid p2 dice");
    let iterations: usize = args[3].parse().expect("Invalid iterations");

    println!("Starting Rust training (Vanilla CFR+) for {}v{} with {} iterations...", p1_dice, p2_dice, iterations);
    
    let start_time = Instant::now();

    // Determine number of threads
    let num_threads = rayon::current_num_threads();
    let iters_per_thread = iterations / num_threads;
    
    println!("Running on {} threads, {} iterations per thread.", num_threads, iters_per_thread);

    // Parallel Map-Reduce
    let final_nodes = (0..num_threads).into_par_iter()
        .map(|_| {
            CFRTrainer::train(p1_dice, p2_dice, iters_per_thread)
        })
        .reduce(HashMap::new, merge_nodes);

    let duration = start_time.elapsed();
    println!("Training complete in {:.2?}", duration);
    println!("Iterations per second: {:.2}", iterations as f64 / duration.as_secs_f64());

    save_strategy(&final_nodes, p1_dice, p2_dice);
}
