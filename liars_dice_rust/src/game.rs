use rand::Rng;
use std::cmp::Ordering;

pub const DICE_FACES: u8 = 6;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    Bid(u8, u8), // Quantity, Face
    Challenge,
}

#[derive(Clone, Debug)]
pub struct GameState {
    pub dice_p1: u8,
    pub dice_p2: u8,
    pub hand_p1: Vec<u8>,
    pub hand_p2: Vec<u8>,
    pub current_bid: Option<(u8, u8)>,
    pub history: Vec<Action>,
    pub current_player: u8, // 0 or 1
}

impl GameState {
    pub fn new(dice_p1: u8, dice_p2: u8) -> Self {
        let mut rng = rand::thread_rng();
        let mut hand_p1 = Vec::with_capacity(dice_p1 as usize);
        let mut hand_p2 = Vec::with_capacity(dice_p2 as usize);

        for _ in 0..dice_p1 {
            hand_p1.push(rng.gen_range(1..=DICE_FACES));
        }
        for _ in 0..dice_p2 {
            hand_p2.push(rng.gen_range(1..=DICE_FACES));
        }
        
        hand_p1.sort();
        hand_p2.sort();

        GameState {
            dice_p1,
            dice_p2,
            hand_p1,
            hand_p2,
            current_bid: None,
            history: Vec::new(),
            current_player: 0,
        }
    }

    pub fn get_valid_actions(&self) -> Vec<Action> {
        let mut actions = Vec::new();
        let total_dice = self.dice_p1 + self.dice_p2;

        if let Some((curr_q, curr_f)) = self.current_bid {
            // 1. Challenge
            actions.push(Action::Challenge);

            // 2. Raise Face
            for f in (curr_f + 1)..=DICE_FACES {
                actions.push(Action::Bid(curr_q, f));
            }

            // 3. Raise Quantity
            for q in (curr_q + 1)..=total_dice {
                for f in 1..=DICE_FACES {
                    actions.push(Action::Bid(q, f));
                }
            }
        } else {
            // First bid
            for q in 1..=total_dice {
                for f in 1..=DICE_FACES {
                    actions.push(Action::Bid(q, f));
                }
            }
        }
        actions
    }

    pub fn apply_action(&mut self, action: Action) -> bool {
        if action == Action::Challenge {
            return true; // Terminal
        }

        if let Action::Bid(q, f) = action {
            self.current_bid = Some((q, f));
        }
        
        self.history.push(action);
        self.current_player = 1 - self.current_player;
        false
    }

    pub fn get_payoff(&self) -> f32 {
        // Payoff for the CHALLENGER (current_player)
        if let Some((bid_q, bid_f)) = self.current_bid {
            let mut count = 0;
            for &d in self.hand_p1.iter().chain(self.hand_p2.iter()) {
                if d == bid_f {
                    count += 1;
                }
            }

            let bidder_wins = count >= bid_q;
            
            if bidder_wins {
                // Bidder (1 - current) wins. Challenger (current) loses.
                -1.0
            } else {
                // Bidder lied. Challenger wins.
                1.0
            }
        } else {
            0.0 // Should not happen
        }
    }

    pub fn get_information_set(&self) -> String {
        let my_hand = if self.current_player == 0 {
            &self.hand_p1
        } else {
            &self.hand_p2
        };

        let hand_str: String = my_hand.iter().map(|d| d.to_string()).collect();
        
        let bid_str = match self.current_bid {
            Some((q, f)) => format!("{}-{}", q, f),
            None => "None".to_string(),
        };

        let count_str = self.history.len().to_string();

        format!("{}|{}|{}", hand_str, bid_str, count_str)
    }
}
