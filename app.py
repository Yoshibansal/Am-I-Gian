import os
import tempfile
import subprocess
import librosa
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create static directory if it doesn't exist
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    """Main karaoke page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    """
    Analyze uploaded audio and return singing score
    Score = (PitchScore * 0.4) + (RhythmScore * 0.3) + (StabilityScore * 0.2) + (LyricsMatchScore * 0.1)
    Each component returns a score from 0-10, final score is 0-10
    """
    uploaded_file_path = None
    wav_file_path = None
    try:
        print("Received analyze request")

        if 'audio' not in request.files:
            print("No audio file in request")
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            print("Empty filename")
            return jsonify({'error': 'No file selected'}), 400

        # Read the file content once to check size and then seek to start
        file_content = audio_file.read()
        if len(file_content) == 0:
            return jsonify({'error': 'Empty audio file'}), 400
        audio_file.seek(0)  # Reset stream position

        file_extension = '.webm' if 'webm' in audio_file.content_type else '.wav'

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            audio_file.save(tmp_file.name)
            uploaded_file_path = tmp_file.name
            print(f"Saved temp file: {uploaded_file_path}")

        wav_file_path = uploaded_file_path.replace(file_extension, ".wav")

        # Convert to WAV using ffmpeg
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", uploaded_file_path, wav_file_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Converted to WAV: {wav_file_path}")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg conversion failed: {e}")
            return jsonify({'error': 'Audio conversion failed. Is ffmpeg installed?'}), 400

        # Load audio using librosa
        y, sr = librosa.load(wav_file_path, sr=22050)
        print(f"Audio loaded: duration={len(y)/sr:.2f}s, sample_rate={sr}")

        if len(y) < sr * 0.5:  # Less than 0.5 seconds
            return jsonify({'error': 'Audio too short (minimum 0.5 seconds)'}), 400

        # 1. PITCH ANALYSIS (40% of score) - returns 0-10
        pitch_score = analyze_pitch(y, sr)
        print(f"Pitch score: {pitch_score}")

        # 2. RHYTHM ANALYSIS (30% of score) - returns 0-10
        rhythm_score = analyze_rhythm(y, sr)
        print(f"Rhythm score: {rhythm_score}")

        # 3. STABILITY ANALYSIS (20% of score) - returns 0-10
        stability_score = analyze_stability(y, sr)
        print(f"Stability score: {stability_score}")

        # 4. LYRICS MATCH SCORE (10% of score) - returns 0-10
        lyrics_score = analyze_lyrics_match(y, sr)
        print(f"Lyrics score: {lyrics_score}")

        # Calculate final score (weighted average)
        final_score = (pitch_score * 0.6) + (rhythm_score * 0.1) + (stability_score * 0.2) + (lyrics_score * 0.1)
        # TODO: change the weightage
        final_score = round(final_score, 1)
        print(f"Final calculated score: {final_score}")

        # Ensure score is between 1-10
        final_score = max(1.0, min(10.0, final_score))

        return jsonify({
            'overallScore': final_score,
            'pitchAccuracy': round(pitch_score * 10, 1),  # Convert to percentage
            'rhythmAccuracy': round(rhythm_score * 10, 1),  # Convert to percentage
            'volumeConsistency': round(stability_score * 10, 1),  # Convert to percentage
            'totalNotes': 100,  # Placeholder
            'correctNotes': round(pitch_score * 10)  # Placeholder based on pitch_score
        })

    except Exception as e:
        print(f"General analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    finally:
        # Ensure temporary files are cleaned up
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            os.unlink(uploaded_file_path)
            print(f"Cleaned up {uploaded_file_path}")
        if wav_file_path and os.path.exists(wav_file_path):
            os.unlink(wav_file_path)
            print(f"Cleaned up {wav_file_path}")

def analyze_pitch(y, sr):
    """
    Analyze pitch accuracy and return score from 0-10
    """
    try:
        # Extract pitch using librosa's pyin algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            # n_thresholds=0.1,
            # threshold=0.1,
            frame_length=2048
        )

        # Filter out nan values and low confidence pitches
        valid_mask = ~np.isnan(f0) & (voiced_probs > 0.3)  # Lower threshold for better detection
        valid_pitches = f0[valid_mask]
        valid_probs = voiced_probs[valid_mask]

        print(f"Valid pitches found: {len(valid_pitches)}")
        
        if len(valid_pitches) == 0:
            print("No valid pitches detected")
            return 1.0  # Minimum score

        # Calculate pitch statistics
        avg_pitch = np.mean(valid_pitches)
        pitch_std = np.std(valid_pitches)
        avg_confidence = np.mean(valid_probs)
        
        print(f"Pitch stats - avg: {avg_pitch:.1f}Hz, std: {pitch_std:.1f}Hz, confidence: {avg_confidence:.2f}")

        # Base score starts at 5 (middle ground)
        score = 5.0

        # Pitch range scoring (human vocal range roughly 80-800Hz)
        if 80 <= avg_pitch <= 800:
            score += 2.0  # Good pitch range
        elif 60 <= avg_pitch <= 1000:
            score += 1.0  # Acceptable range
        else:
            score -= 1.0  # Outside normal range

        # Pitch stability scoring (lower std = more stable)
        if pitch_std < 20:
            score += 2.0  # Very stable
        elif pitch_std < 40:
            score += 1.5  # Good stability
        elif pitch_std < 60:
            score += 1.0  # Moderate stability
        elif pitch_std < 80:
            score += 0.5  # Some instability
        else:
            score -= 0.5  # Very unstable

        # Confidence scoring
        if avg_confidence > 0.8:
            score += 1.0  # High confidence
        elif avg_confidence > 0.6:
            score += 0.5  # Medium confidence
        elif avg_confidence < 0.4:
            score -= 0.5  # Low confidence

        # Ensure score is between 0-10
        score = max(0.0, min(10.0, score))
        
        return score

    except Exception as e:
        print(f"Pitch analysis error: {e}")
        return 3.0  # Default middle-low score on error

def analyze_rhythm(y, sr):
    """
    Analyze rhythm consistency and return score from 0-10
    """
    try:
        # Extract tempo and beat information
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
        tempo = float(tempo)
        
        print(f"Detected tempo: {tempo:.1f} BPM, beats: {len(beats)}")
        
        if len(beats) < 3:
            print("Not enough beats detected")
            return 2.0  # Low score for insufficient rhythm detection

        # Calculate beat intervals (time between beats)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        beat_intervals = np.diff(beat_times)
        
        if len(beat_intervals) == 0:
            return 2.0
        
        # Calculate rhythm consistency
        mean_interval = float(np.mean(beat_intervals))
        std_interval = float(np.std(beat_intervals))
        
        print(f"Beat interval stats - mean: {mean_interval:.3f}s, std: {std_interval:.3f}s")
        
        # Avoid division by zero
        if mean_interval == 0:
            rhythm_consistency = 0.0
        else:
            # Coefficient of variation (lower = more consistent)
            cv = std_interval / mean_interval
            rhythm_consistency = max(0, 1.0 - cv)
        
        print(f"Rhythm consistency: {rhythm_consistency:.3f}")
        
        # Base score starts at 4
        score = 4.0
        
        # Tempo scoring (most songs are between 60-180 BPM)
        if 60 <= tempo <= 180:
            score += 2.0  # Good tempo range
        elif 40 <= tempo <= 200:
            score += 1.0  # Acceptable range
        else:
            score -= 1.0  # Outside normal range
        
        # Rhythm consistency scoring
        if rhythm_consistency > 0.8:
            score += 3.0  # Very consistent
        elif rhythm_consistency > 0.6:
            score += 2.0  # Good consistency
        elif rhythm_consistency > 0.4:
            score += 1.0  # Moderate consistency
        elif rhythm_consistency > 0.2:
            score += 0.5  # Some consistency
        else:
            score -= 0.5  # Poor consistency
        
        # Number of beats bonus
        if len(beats) > 20:
            score += 1.0  # Good rhythmic content
        elif len(beats) > 10:
            score += 0.5  # Moderate rhythmic content
        
        # Ensure score is between 0-10
        score = max(0.0, min(10.0, score))
        
        return score

    except Exception as e:
        print(f"Rhythm analysis error: {e}")
        return 3.0  # Default middle-low score on error

def analyze_stability(y, sr):
    """
    Analyze audio stability (volume and spectral consistency) and return score from 0-10
    """
    try:
        # Calculate spectral centroids and RMS energy
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        
        print(f"Stability analysis - RMS mean: {np.mean(rms):.6f}, Spectral centroid mean: {np.mean(spectral_centroids):.1f}")
        
        # Check for very low energy
        if np.mean(rms) < 0.001:
            print("Very low energy detected")
            return 1.0  # Very low energy
        
        # Calculate stability metrics
        mean_centroid = np.mean(spectral_centroids)
        mean_rms = np.mean(rms)
        
        # Avoid division by zero
        if mean_centroid == 0:
            centroid_stability = 0.0
        else:
            centroid_cv = np.std(spectral_centroids) / mean_centroid
            centroid_stability = max(0, 1.0 - centroid_cv)
        
        if mean_rms == 0:
            rms_stability = 0.0
        else:
            rms_cv = np.std(rms) / mean_rms
            rms_stability = max(0, 1.0 - rms_cv)
        
        print(f"Stability metrics - centroid: {centroid_stability:.3f}, RMS: {rms_stability:.3f}")
        
        # Base score starts at 4
        score = 4.0
        
        # Volume consistency scoring
        if rms_stability > 0.7:
            score += 2.5  # Very consistent volume
        elif rms_stability > 0.5:
            score += 2.0  # Good volume consistency
        elif rms_stability > 0.3:
            score += 1.5  # Moderate consistency
        elif rms_stability > 0.1:
            score += 1.0  # Some consistency
        else:
            score += 0.5  # Poor consistency
        
        # Spectral consistency scoring
        if centroid_stability > 0.7:
            score += 2.5  # Very consistent tone
        elif centroid_stability > 0.5:
            score += 2.0  # Good tonal consistency
        elif centroid_stability > 0.3:
            score += 1.5  # Moderate consistency
        elif centroid_stability > 0.1:
            score += 1.0  # Some consistency
        else:
            score += 0.5  # Poor consistency
        
        # Energy level bonus/penalty
        if 0.01 <= mean_rms <= 0.5:
            score += 1.0  # Good energy level
        elif 0.005 <= mean_rms <= 0.8:
            score += 0.5  # Acceptable energy level
        else:
            score -= 0.5  # Too quiet or too loud
        
        # Ensure score is between 0-10
        score = max(0.0, min(10.0, score))
        
        return score

    except Exception as e:
        print(f"Stability analysis error: {e}")
        return 3.0  # Default middle-low score on error

def analyze_lyrics_match(y, sr):
    """
    Analyze vocal characteristics that might indicate lyrics matching and return score from 0-10
    This is a simplified approach since actual lyrics matching requires speech recognition
    """
    try:
        # Calculate zero crossing rate (indicates speech-like characteristics)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]
        avg_zcr = np.mean(zcr)
        
        # Calculate spectral rolloff (indicates vocal characteristics)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        avg_rolloff = np.mean(spectral_rolloff)
        
        # Calculate MFCCs (vocal characteristics)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        print(f"Lyrics analysis - ZCR: {avg_zcr:.4f}, Rolloff: {avg_rolloff:.1f}Hz")
        
        # Base score starts at 5
        score = 5.0
        
        # Zero crossing rate scoring (speech-like characteristics)
        if 0.05 <= avg_zcr <= 0.15:
            score += 2.0  # Good speech-like characteristics
        elif 0.02 <= avg_zcr <= 0.20:
            score += 1.5  # Acceptable speech characteristics
        elif 0.01 <= avg_zcr <= 0.25:
            score += 1.0  # Some speech characteristics
        else:
            score += 0.5  # Poor speech characteristics
        
        # Spectral rolloff scoring (vocal range)
        if 2000 <= avg_rolloff <= 8000:
            score += 1.5  # Good vocal range
        elif 1000 <= avg_rolloff <= 10000:
            score += 1.0  # Acceptable vocal range
        else:
            score += 0.5  # Outside typical vocal range
        
        # MFCC-based vocal quality (simplified)
        if -20 <= mfcc_mean[0] <= 20:  # First MFCC coefficient
            score += 1.5  # Good vocal quality
        else:
            score += 0.5  # Poor vocal quality
        
        # Ensure score is between 0-10
        score = max(0.0, min(10.0, score))
        
        return score

    except Exception as e:
        print(f"Lyrics analysis error: {e}")
        return 5.0  # Default middle score on error

if __name__ == '__main__':
    # app.run(debug=True, port=5000)
    app.run()