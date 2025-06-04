import numpy as np
import pandas as pd
import random

class GamblingBehaviorSimulator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_user_session(self, user_type='normal', session_length_minutes=60):
        """Generate a single gambling session"""
        
        # Simple behavioral parameters
        params = {
            'normal': {
                'bet_frequency': (30, 120),
                'bet_amount_base': (5, 25),
                'loss_chase_prob': 0.1
            },
            'at_risk': {
                'bet_frequency': (10, 60),
                'bet_amount_base': (10, 50),
                'loss_chase_prob': 0.4
            },
            'high_risk': {
                'bet_frequency': (2, 15),
                'bet_amount_base': (20, 100),
                'loss_chase_prob': 0.8
            }
        }
        
        p = params[user_type]
        session_data = []
        current_time = 0
        balance = 1000
        consecutive_losses = 0
        
        max_time = session_length_minutes * 60  # seconds
        
        while current_time < max_time and len(session_data) < 100:  # limit to 100 bets
            # Time between bets
            time_gap = np.random.uniform(p['bet_frequency'][0], p['bet_frequency'][1])
            current_time += time_gap
            
            # Bet amount
            base_bet = np.random.uniform(p['bet_amount_base'][0], p['bet_amount_base'][1])
            
            # Loss chasing: increase bet after losses
            if consecutive_losses > 0 and random.random() < p['loss_chase_prob']:
                bet_amount = base_bet * (1 + consecutive_losses * 0.5)
            else:
                bet_amount = base_bet
            
            bet_amount = min(bet_amount, balance * 0.5)  # don't bet more than half balance
            
            # Outcome (47% win probability - house edge)
            if random.random() < 0.47:
                outcome = bet_amount * 1.8  # win
                consecutive_losses = 0
            else:
                outcome = -bet_amount  # loss
                consecutive_losses += 1
            
            balance += outcome
            
            session_data.append({
                'timestamp': current_time,
                'bet_amount': bet_amount,
                'outcome': outcome,
                'balance': balance,
                'consecutive_losses': consecutive_losses,
                'user_type': user_type
            })
            
            if balance < 10:  # stop if almost broke
                break
        
        return session_data

# Simple loss chasing calculation
def simple_loss_chasing(user_data):
    """Simple loss chasing score"""
    chase_events = 0
    for i in range(1, len(user_data)):
        if (user_data.iloc[i-1]['outcome'] < 0 and  # previous was loss
            user_data.iloc[i]['bet_amount'] > user_data.iloc[i-1]['bet_amount'] * 1.2):  # bet increased 20%
            chase_events += 1
    return chase_events / len(user_data) if len(user_data) > 0 else 0

# Generate 1000 users
print("🎲 Generating 1000 gambling users...")

simulator = GamblingBehaviorSimulator()

# Realistic distribution: 70% normal, 25% at-risk, 5% high-risk
user_types = (['normal'] * 700 + ['at_risk'] * 250 + ['high_risk'] * 50)

all_users = []
for user_id in range(1000):
    user_type = user_types[user_id]
    
    # Random session length by risk level
    if user_type == 'normal':
        session_length = np.random.uniform(15, 60)  # 15-60 minutes
    elif user_type == 'at_risk':
        session_length = np.random.uniform(30, 120)  # 30-120 minutes  
    else:  # high_risk
        session_length = np.random.uniform(60, 180)  # 1-3 hours
    
    session = simulator.generate_user_session(user_type, session_length)
    for bet in session:
        bet['user_id'] = user_id
    all_users.extend(session)
    
    # Progress indicator
    if (user_id + 1) % 100 == 0:
        print(f"  Generated {user_id + 1}/1000 users...")

df = pd.DataFrame(all_users)

print(f"✅ Generated {len(df)} total bets for 1000 users")
print(f"📊 User distribution: {df['user_type'].value_counts().to_dict()}")
print(f"📈 Average bets per user: {len(df)/1000:.1f}")

# Calculate features for each user
print("\n🧠 Calculating features for each user...")
user_features = []

for user_id in range(1000):
    user_data = df[df['user_id'] == user_id].copy()
    if len(user_data) == 0:
        continue
    
    # Loss chasing score
    chase_score = simple_loss_chasing(user_data)
    
    # Session intensity (bets per minute)
    session_duration_min = (user_data['timestamp'].max() - user_data['timestamp'].min()) / 60
    session_intensity = len(user_data) / max(session_duration_min, 1)  # avoid division by zero
    
    # Average bet size
    avg_bet = user_data['bet_amount'].mean()
    
    # Consecutive losses (max streak)
    max_losses = user_data['consecutive_losses'].max()
    
    user_features.append({
        'user_id': user_id,
        'user_type': user_data['user_type'].iloc[0],
        'loss_chasing_score': chase_score,
        'session_intensity': session_intensity,
        'avg_bet_size': avg_bet,
        'max_loss_streak': max_losses,
        'total_bets': len(user_data),
        'session_duration_min': session_duration_min
    })

features_df = pd.DataFrame(user_features)

# Create risk labels for ML
risk_mapping = {'normal': 0, 'at_risk': 1, 'high_risk': 2}
features_df['risk_label'] = features_df['user_type'].map(risk_mapping)

# Analyze results by user type
print("\n📊 Feature Analysis by Risk Level:")
for user_type in ['normal', 'at_risk', 'high_risk']:
    subset = features_df[features_df['user_type'] == user_type]
    print(f"\n{user_type.upper()} (n={len(subset)}):")
    print(f"  Loss Chasing: {subset['loss_chasing_score'].mean():.3f} ± {subset['loss_chasing_score'].std():.3f}")
    print(f"  Session Intensity: {subset['session_intensity'].mean():.3f} ± {subset['session_intensity'].std():.3f}")
    print(f"  Avg Bet Size: ${subset['avg_bet_size'].mean():.2f} ± ${subset['avg_bet_size'].std():.2f}")
    print(f"  Max Loss Streak: {subset['max_loss_streak'].mean():.1f} ± {subset['max_loss_streak'].std():.1f}")

print(f"\n💾 Dataset Summary:")
print(f"  Total Users: {len(features_df)}")
print(f"  Total Gambling Sessions: {len(df)}")
print(f"  Features per User: {len(features_df.columns) - 2}")  # exclude user_id and user_type

print("\n🎯 SUCCESS! You now have:")
print("✅ 1000 synthetic gambling users")
print("✅ Behavioral features extracted") 
print("✅ Risk labels for machine learning")
print("✅ Real data ready for your first ML model!")

# Save data for later use
features_df.to_csv('gambling_features.csv', index=False)
df.to_csv('gambling_sessions.csv', index=False)
print("\n💾 Data saved to CSV files for analysis")