#!/bin/bash
# Quick test script for CNN model

echo "======================================"
echo "CNN Model Test Script"
echo "======================================"
echo ""

# Test model implementation
echo "1. Testing CNN model architecture..."
python3 -m models.cnn.cnn_policy

if [ $? -eq 0 ]; then
    echo "✓ CNN model tests passed!"
else
    echo "✗ CNN model tests failed!"
    exit 1
fi

echo ""
echo "======================================"
echo "All CNN tests completed successfully!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Train a model: python training/train_cnn.py --debug"
echo "  2. Evaluate model: python test_cnn.py --checkpoint <path>"
echo "  3. See docs/CNN-TRAINING-GUIDE.md for full guide"
