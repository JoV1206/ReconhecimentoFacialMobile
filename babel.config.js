module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    // O plugin do worklets-core precisa ser o ultimo da lista: ele transforma as
    // funcoes marcadas com a diretiva 'worklet' para rodarem no runtime paralelo
    // usado pelos frame processors da VisionCamera.
    //
    // Ele chama internamente @babel/plugin-proposal-{optional-chaining,
    // nullish-coalescing-operator} e mais tres transforms, mas nao os declara como
    // dependencia — por isso eles aparecem em devDependencies, fixados em ^7 (as
    // versoes 8 exigem @babel/core 8 e conflitam com o toolchain do Expo 54).
    plugins: ['react-native-worklets-core/plugin'],
  };
};
