-- ============================================================================
-- Schéma de la base TRADEBOT
-- ============================================================================
-- Corrections par rapport à database.sql :
--   1. Ajout d'un index UNIQUE (symbol, time) sur training_data_hour.
--      Sans lui, relancer une collecte dupliquait tout l'historique — et un
--      historique dupliqué fausse SILENCIEUSEMENT tous les indicateurs.
--   2. DECIMAL(18,6) au lieu de FLOAT : le FLOAT est approximatif, ce qui est
--      inacceptable pour des prix (erreurs d'arrondi cumulées).
--   3. Index sur (symbol, time) : la requête de chargement passe d'un scan
--      complet de table à une lecture indexée.
--   4. Colonnes du live alignées sur ce que le bot calcule réellement
--      (rsi, adx, stoch_k, stoch_d) — l'ancienne table avait osc/osc_ema
--      alors que le code insérait autre chose.
-- ============================================================================

CREATE DATABASE IF NOT EXISTS TRADEBOT
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE TRADEBOT;

-- Historique utilisé pour l'entraînement -------------------------------------
CREATE TABLE IF NOT EXISTS training_data_hour (
    id          BIGINT PRIMARY KEY AUTO_INCREMENT,
    time        DATETIME       NOT NULL,
    symbol      VARCHAR(20)    NOT NULL,
    openPrice   DECIMAL(18,6),
    highPrice   DECIMAL(18,6),
    lowPrice    DECIMAL(18,6),
    closePrice  DECIMAL(18,6),
    volume      DECIMAL(20,2),

    -- Idempotence de la collecte : une bougie = une ligne, définitivement.
    UNIQUE KEY uq_symbol_time (symbol, time),
    KEY idx_symbol_time (symbol, time)
) ENGINE=InnoDB;

-- Journal du bot en temps réel ------------------------------------------------
CREATE TABLE IF NOT EXISTS stocks (
    id             BIGINT PRIMARY KEY AUTO_INCREMENT,
    time           TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,
    symbol         VARCHAR(20)    NOT NULL,
    currentPrice   DECIMAL(18,6),
    percentChange  DECIMAL(10,4),
    changePrice    DECIMAL(18,6),
    highPriceDay   DECIMAL(18,6),
    lowPriceDay    DECIMAL(18,6),
    volume         DECIMAL(20,2),
    sma            DECIMAL(18,6),
    ema            DECIMAL(18,6),
    rsi            DECIMAL(10,4),
    adx            DECIMAL(10,4),
    stoch_k        DECIMAL(10,4),
    stoch_d        DECIMAL(10,4),

    KEY idx_symbol_time (symbol, time)
) ENGINE=InnoDB;