// Learn more https://docs.expo.io/guides/customizing-metro
const { getDefaultConfig } = require('expo/metro-config');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// O modelo MobileFaceNet e carregado via require('...tflite'), entao o Metro
// precisa tratar .tflite como asset binario e nao tentar interpretar como codigo.
config.resolver.assetExts.push('tflite');

module.exports = config;
